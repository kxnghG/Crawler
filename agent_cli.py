#!/usr/bin/env python3
import os
import re
import io
import sys
import json
import time
import argparse
import contextlib
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except Exception:
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import tldextract
except Exception:
    tldextract = None

try:
    from jsonschema import Draft7Validator, FormatChecker
except Exception:
    Draft7Validator = None
    FormatChecker = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

def _load_env_vars() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, ".env")
    example_path = os.path.join(here, ".env.example")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    cur = os.environ.get("GEMINI_API_KEY")
    if (cur is None or cur.strip() == "") and os.path.exists(example_path):
        load_dotenv(example_path, override=True)

try:
    from crawl4ai import AsyncWebCrawler
except Exception:
    AsyncWebCrawler = None


SOCIAL_PLATFORMS = {
    "twitter": ["twitter.com", "x.com"],
    "x": ["x.com"],
    "facebook": ["facebook.com"],
    "instagram": ["instagram.com"],
    "linkedin": ["linkedin.com"],
    "youtube": ["youtube.com", "youtu.be"],
    "tiktok": ["tiktok.com"],
    "pinterest": ["pinterest.com"],
    "github": ["github.com"],
}

SYSTEM_QA = (
    "You are a precise Q&A agent. Answer ONLY from the provided JSON document. "
    "If data is missing, say you don't have that information. Be concise."
)


def extract_domain(value: str) -> str:
    value = value.strip()
    if "@" in value:
        m = re.search(r"@([A-Za-z0-9.-]+)", value)
        if not m:
            raise ValueError("Invalid email")
        return m.group(1).lower()
    value = value.removeprefix("https://").removeprefix("http://")
    value = value.removeprefix("www.")
    return value.split("/")[0].lower()


def resolve_base_url(domain: str) -> str:
    candidates = [f"https://{domain}", f"https://www.{domain}", f"http://{domain}", f"http://www.{domain}"]
    if not requests:
        return candidates[0]
    for url in candidates:
        try:
            r = requests.get(url, timeout=10, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code < 400:
                return r.url
        except Exception:
            continue
    return candidates[0]


def fetch_html(url: str) -> str:
    if AsyncWebCrawler is not None:
        try:
            import asyncio

            async def run() -> str:
                async with AsyncWebCrawler() as crawler:
                    res = await crawler.arun(url=url)
                    return getattr(res, "html", None) or getattr(res, "markdown", "") or ""

            return asyncio.run(run())
        except Exception:
            pass
    if not requests:
        return ""
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code < 400:
            return r.text
    except Exception:
        return ""
    return ""


def parse_meta(html: str) -> Dict[str, Any]:
    if not BeautifulSoup:
        return {}
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
    desc = desc_tag.get("content").strip() if desc_tag and desc_tag.get("content") else None
    gen_tag = soup.find("meta", attrs={"name": "generator"})
    generator = gen_tag.get("content").strip() if gen_tag and gen_tag.get("content") else None
    tagline = None
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True) and (not title or h1.get_text(strip=True) != title):
        tagline = h1.get_text(strip=True)
    return {"title": title, "description": desc, "tagline": tagline, "generator": generator}


def _is_platform_profile_url(platform: str, href: str) -> bool:
    h = href.lower()
    if platform in {"twitter", "x"}:
        return ("twitter.com/" in h or "x.com/" in h) and not any(p in h for p in ["/intent/", "/share", "/status/"])
    if platform == "facebook":
        return "facebook.com/" in h and not any(p in h for p in ["/share.php", "/dialog/"])
    if platform == "instagram":
        return "instagram.com/" in h and "/p/" not in h
    if platform == "linkedin":
        return "linkedin.com/" in h and any(p in h for p in ["/company/", "/school/", "/in/"])
    if platform == "youtube":
        return "youtube.com/" in h or "youtu.be/" in h
    return True


def parse_social_links(base_url: str, html: str) -> List[Dict[str, Any]]:
    if not BeautifulSoup or not tldextract:
        return []
    soup = BeautifulSoup(html, "lxml")
    profiles: List[Dict[str, Any]] = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("mailto:"):
            continue
        if href.startswith("/"):
            href = base_url.rstrip("/") + href
        if not href.startswith("http"):
            continue
        try:
            host = tldextract.extract(href)
            reg = getattr(host, "top_domain_under_public_suffix", None) or getattr(host, "registered_domain", None)
            host_name = reg or ".".join([x for x in [host.domain, host.suffix] if x])
        except Exception:
            continue
        for platform, domains in SOCIAL_PLATFORMS.items():
            if any(str(host_name).endswith(d) for d in domains):
                key = (platform, href)
                if key in seen:
                    break
                # Filter loose enough to include useful links (esp. YouTube watch/embed)
                if not _is_platform_profile_url(platform, href):
                    break
                seen.add(key)
                # Derive a simple handle
                handle: Optional[str] = None
                try:
                    from urllib.parse import urlparse
                    p = urlparse(href)
                    parts = [x for x in p.path.split("/") if x]
                    if platform in {"twitter", "x", "instagram"} and parts:
                        handle = parts[0]
                    elif platform == "facebook" and parts:
                        handle = parts[0]
                    elif platform == "linkedin" and parts:
                        # e.g., /company/<handle> or /school/<handle>
                        handle = parts[-1] or (parts[-2] if len(parts) > 1 else None)
                    elif platform == "youtube" and parts:
                        # categorize by first segment (channel|watch|embed|@handle)
                        handle = parts[0]
                    elif platform == "github" and parts:
                        handle = parts[0]
                except Exception:
                    handle = None
                item = {"platform": platform, "url": href}
                if handle:
                    item["handle"] = handle
                profiles.append(item)
                break
    return profiles


def parse_contacts(base_url: str, html: str) -> List[Dict[str, Any]]:
    """Extract emails and phone numbers as Contact items.
    Kind detection is heuristic based on link text or URL.
    """
    if not BeautifulSoup:
        return []
    soup = BeautifulSoup(html, "lxml")
    contacts: List[Dict[str, Any]] = []
    seen: set = set()

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        text = a.get_text(" ", strip=True).lower()
        if href.startswith("mailto:"):
            email = href.split(":", 1)[1]
            kind = "other"
            if any(k in (text or href).lower() for k in ["sales", "business"]):
                kind = "sales"
            elif any(k in (text or href).lower() for k in ["support", "help", "care"]):
                kind = "support"
            elif "press" in (text or href).lower():
                kind = "press"
            elif any(k in (text or href).lower() for k in ["partner", "partnership"]):
                kind = "partnerships"
            key = (kind, email)
            if key in seen:
                continue
            seen.add(key)
            contacts.append({"kind": kind, "email": email})

    for a in soup.find_all("a", href=True):
        href = (a.get("href") or "").strip()
        if href.startswith("tel:"):
            phone = href.split(":", 1)[1]
            key = ("other", phone)
            if key in seen:
                continue
            seen.add(key)
            contacts.append({"kind": "other", "phone": phone})

    return contacts


def parse_schema_org(html: str) -> List[Dict[str, Any]]:
    if not BeautifulSoup:
        return []
    blocks: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script", attrs={"type": re.compile(r"application/(ld\+)?json", re.I)}):
        try:
            txt = script.string or script.text or ""
            if not txt:
                continue
            data = json.loads(txt)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        blocks.append(item)
            elif isinstance(data, dict):
                blocks.append(data)
        except Exception:
            continue
    return blocks


def extract_addresses_from_schema(schema_blocks: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Extract (hq, locations, industry) from schema.org blocks when present."""
    def norm_address(addr: Any) -> Optional[Dict[str, Any]]:
        if isinstance(addr, str):
            return {"street": addr}
        if isinstance(addr, dict):
            return {
                **({"street": addr.get("streetAddress")} if addr.get("streetAddress") else {}),
                **({"city": addr.get("addressLocality")} if addr.get("addressLocality") else {}),
                **({"state": addr.get("addressRegion")} if addr.get("addressRegion") else {}),
                **({"postal_code": addr.get("postalCode")} if addr.get("postalCode") else {}),
                **({"country": addr.get("addressCountry")} if addr.get("addressCountry") else {}),
            }
        return None

    hq: Optional[Dict[str, Any]] = None
    locations: List[Dict[str, Any]] = []
    industry: Optional[str] = None
    for b in schema_blocks:
        btype = b.get("@type") if isinstance(b, dict) else None
        if isinstance(btype, list) and btype:
            btype = next((t for t in btype if isinstance(t, str)), btype[0])
        if isinstance(btype, str) and not industry:
            industry = btype.lower()
        # Address candidates
        addr_candidates: List[Any] = []
        if isinstance(b.get("address"), list):
            addr_candidates.extend(b.get("address", []))
        elif b.get("address") is not None:
            addr_candidates.append(b.get("address"))
        loc = b.get("location")
        if isinstance(loc, dict) and loc.get("address") is not None:
            addr_candidates.append(loc.get("address"))
        for c in addr_candidates:
            a = norm_address(c)
            if not a:
                continue
            if not hq:
                hq = a
            if a not in locations:
                locations.append(a)
    return hq, locations, industry


def detect_tech_stack(base_url: str, html: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not BeautifulSoup:
        return items
    soup = BeautifulSoup(html, "lxml")
    meta = soup.find("meta", attrs={"name": "generator"})
    if meta and meta.get("content"):
        cont = meta.get("content")
        name = None
        version = None
        if "wordpress" in cont.lower():
            name = "WordPress"
            m = re.search(r"\b(\d+\.\d+(?:\.\d+)?)\b", cont)
            version = m.group(1) if m else None
            items.append({"name": name, "category": "cms", **({"version": version} if version else {}), "detected_from": "meta[name=generator]"})
        else:
            items.append({"name": cont.strip(), "category": "generator", "detected_from": "meta[name=generator]"})

    def add(name: str, category: str, detected_from: str, version: Optional[str] = None):
        item = {"name": name, "category": category, "detected_from": detected_from}
        if version:
            item["version"] = version
        items.append(item)

    for tag in soup.find_all(["script", "link" ]):
        src = tag.get("src") or tag.get("href") or ""
        if not src:
            continue
        s = src.lower()
        if "googletagmanager.com" in s:
            add("Google Tag Manager", "analytics", src)
        if "gtag/js" in s or "analytics.js" in s:
            add("Google Analytics", "analytics", src)
        if "hotjar" in s:
            add("Hotjar", "analytics", src)
        if "hs-analytics" in s or "hubspot" in s:
            add("HubSpot", "marketing", src)
        if "shopify" in s:
            add("Shopify", "commerce", src)
        if "cdn-cgi" in s:
            add("Cloudflare", "cdn", src)
        if "bootstrap" in s:
            add("Bootstrap", "ui", src)

    if soup.find(id="__NEXT_DATA__"):
        add("Next.js", "framework", "#__NEXT_DATA__")
    return items


def collect_site_variants_and_microsites(domain: str, base_url: str, html: str) -> Tuple[List[str], List[str]]:
    """Return alternates (scheme/www variants) and microsites (subdomain roots discovered on-page)."""
    alternates = [
        f"https://{domain}",
        f"http://{domain}",
        f"https://www.{domain}",
        f"http://www.{domain}",
    ]
    if base_url not in alternates:
        alternates.append(base_url)
    microsites: List[str] = []
    seen: set = set()
    if BeautifulSoup and tldextract:
        soup = BeautifulSoup(html, "lxml")
        from urllib.parse import urljoin
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            if href.startswith("/"):
                href = urljoin(base_url, href)
            if not href.startswith("http"):
                continue
            try:
                ext = tldextract.extract(href)
                reg = getattr(ext, "top_domain_under_public_suffix", None) or getattr(ext, "registered_domain", None)
                if not reg or not reg.endswith(domain):
                    continue
                host = ".".join([p for p in [ext.subdomain, ext.domain, ext.suffix] if p])
                # Canonical site root
                site = f"{href.split('://',1)[0]}://{host}/"
                if ext.subdomain and ext.subdomain not in {"www", ""}:
                    key = (host, site)
                    if key in seen:
                        continue
                    seen.add(key)
                    microsites.append(site.rstrip("/"))
            except Exception:
                continue
    # de-dup
    def dedup(seq: List[str]) -> List[str]:
        out: List[str] = []
        s: set = set()
        for x in seq:
            if x in s:
                continue
            s.add(x)
            out.append(x)
        return out
    return dedup(alternates), dedup(microsites)


def find_sitemaps(base_url: str) -> List[str]:
    urls: List[str] = []
    if not requests:
        return urls
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(base_url.rstrip("/") + "/robots.txt", timeout=10, headers=headers)
        if r.status_code < 400:
            for line in r.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    u = line.split(":", 1)[1].strip()
                    if u:
                        urls.append(u)
    except Exception:
        pass
    for path in ["/sitemap.xml", "/sitemap_index.xml"]:
        candidate = base_url.rstrip("/") + path
        try:
            r = requests.get(candidate, timeout=10, headers=headers)
            if r.status_code < 400:
                urls.append(candidate)
        except Exception:
            continue
    seen = set()
    result: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        result.append(u)
    return result


def build_json(domain: str, url: str, html: str) -> Dict[str, Any]:
    meta = parse_meta(html)
    social = parse_social_links(url, html)
    contacts = parse_contacts(url, html)
    schema_org = parse_schema_org(html)
    sitemaps = find_sitemaps(url)
    tech_stack = detect_tech_stack(url, html)
    hq, locations, industry = extract_addresses_from_schema(schema_org)
    alternates, microsites = collect_site_variants_and_microsites(domain, url, html)

    legal_name = None
    # Prefer schema.org Organization/CollegeOrUniversity/LocalBusiness name if present
    for block in schema_org:
        try:
            btype = block.get("@type") if isinstance(block, dict) else None
            if isinstance(btype, list):
                btype = next((t for t in btype if isinstance(t, str)), None)
            if isinstance(block, dict) and btype in {"Organization", "CollegeOrUniversity", "LocalBusiness", "EducationalOrganization"}:
                nm = block.get("name")
                if isinstance(nm, str) and nm.strip():
                    legal_name = nm.strip()
                    break
        except Exception:
            continue
    if meta.get("title"):
        # Heuristic: take title up to a pipe or dash
        legal_name = legal_name or (re.split(r"[\-|–|—]", meta["title"])[0].strip() or None)
    if not legal_name:
        legal_name = domain.split(".")[0]

    company: Dict[str, Any] = {
        "legal_name": legal_name,
        **({"description": meta["description"]} if meta.get("description") else {}),
        **({"tagline": meta["tagline"]} if meta.get("tagline") else {}),
        **({"contacts": contacts} if contacts else {}),
        **({"hq": hq} if hq else {}),
        **({"locations": locations} if locations else {}),
        **({"industry": industry} if industry else {}),
    }

    websites: Dict[str, Any] = {
        "primary": url,
        **({"alternates": alternates} if alternates else {}),
        **({"microsites": microsites} if microsites else {}),
        **({"sitemaps": sitemaps} if sitemaps else {}),
        **({"tech_stack": tech_stack} if tech_stack else {}),
        **({"schema_org": schema_org} if schema_org else {}),
    }

    provenance: List[Dict[str, Any]] = []
    captured_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if meta.get("title"):
        provenance.append({
            "json_pointer": "/Company/legal_name",
            "source_url": url,
            "selector": "title",
            "method": "css",
            "confidence": 0.6,
            "captured_at": captured_at,
        })
    if meta.get("description"):
        provenance.append({
            "json_pointer": "/Company/description",
            "source_url": url,
            "selector": "meta[name=description]",
            "method": "css",
            "confidence": 0.7,
            "captured_at": captured_at,
        })
    if social:
        provenance.append({
            "json_pointer": "/SocialProfiles",
            "source_url": url,
            "selector": "a[href]",
            "method": "css",
            "confidence": 0.7,
            "captured_at": captured_at,
        })
    if schema_org:
        provenance.append({
            "json_pointer": "/Websites/schema_org",
            "source_url": url,
            "selector": "script[type=application/ld+json]",
            "method": "css",
            "confidence": 0.8,
            "captured_at": captured_at,
        })
    if sitemaps:
        provenance.append({
            "json_pointer": "/Websites/sitemaps",
            "source_url": url.rstrip("/") + "/robots.txt",
            "selector": "Sitemap:",
            "method": "api",
            "confidence": 0.9,
            "captured_at": captured_at,
        })
    if alternates:
        provenance.append({
            "json_pointer": "/Websites/alternates",
            "source_url": url,
            "method": "llm",
            "confidence": 0.4,
            "captured_at": captured_at,
        })
    if microsites:
        provenance.append({
            "json_pointer": "/Websites/microsites",
            "source_url": url,
            "selector": "a[href]",
            "method": "css",
            "confidence": 0.7,
            "captured_at": captured_at,
        })
    if hq:
        provenance.append({
            "json_pointer": "/Company/hq",
            "source_url": url,
            "method": "llm",
            "confidence": 0.5,
            "captured_at": captured_at,
        })
    if locations:
        provenance.append({
            "json_pointer": "/Company/locations",
            "source_url": url,
            "method": "llm",
            "confidence": 0.5,
            "captured_at": captured_at,
        })

    # Simple heuristic assessment scores to approximate presence (0-100)
    def compute_assessment_scores(company: Dict[str, Any], websites: Dict[str, Any], social: List[Dict[str, Any]], data_snapshot: Dict[str, Any]) -> Dict[str, int]:
        def clamp(x: int) -> int:
            return max(0, min(100, x))
        brand_presence = clamp(20 + 10 * min(len(social), 6) + (20 if websites.get("schema_org") else 0))
        products_services = clamp(10 * (len(data_snapshot.get("Products", [])) > 0) + 10 * (len(data_snapshot.get("Services", [])) > 0))
        seo_authority = clamp(20 + 10 * len(websites.get("sitemaps", [])) + 5 * len(websites.get("tech_stack", [])))
        social_community = clamp(15 * min(len(social), 6))
        reputation_reviews = clamp(10 * min(len(data_snapshot.get("ReviewAggregates", [])), 6))
        talent_culture = clamp(10 * min(len(data_snapshot.get("EmployerRatings", [])), 5) + (10 if len(data_snapshot.get("JobListings", [])) > 0 else 0))
        # overall as average of available categories
        parts = [brand_presence, products_services, seo_authority, social_community, reputation_reviews, talent_culture]
        overall = int(sum(parts) / len(parts))
        return {
            "brand_presence": brand_presence,
            "products_services": products_services,
            "seo_authority": seo_authority,
            "social_community": social_community,
            "reputation_reviews": reputation_reviews,
            "talent_culture": talent_culture,
            "overall": overall,
        }

    data: Dict[str, Any] = {
        "Company": company,
        "Websites": websites,
        "SocialProfiles": social,
        "Products": [],
        "Services": [],
        "GoogleBusinessProfile": None,
        "KnowledgePanel": None,
        "Rankings": [],
        "BacklinkMetrics": None,
        "ReviewAggregates": [],
        "JobListings": [],
        "EmployerRatings": [],
        "SentimentSamples": [],
        "MediaMentions": [],
        "Events": [],
        "MediaAppearances": [],
        "TechHealthReport": None,
        "FundingRounds": [],
        "OpenSourceRepos": [],
        "Chatbots": [],
        "Patents": [],
        "Publications": [],
        "Mentions": [],
        "AssessmentScores": None,
        "Provenance": provenance,
    "captured_at": captured_at,
    }
    # Compute basic AssessmentScores based on current snapshot
    try:
        data["AssessmentScores"] = compute_assessment_scores(company, websites, social, data)
    except Exception:
        pass
    return data


def load_schema_defs() -> Optional[Dict[str, Any]]:
    """Load component schemas from crawlers.json (repo root)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "crawlers.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def validate_output(data: Dict[str, Any], defs: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    if not defs or Draft7Validator is None:
        return True, []
    fmt = FormatChecker() if FormatChecker else None
    errors: List[str] = []

    def with_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
        # Embed all component definitions under $defs so internal $ref lookups resolve
        return {"$defs": defs, **schema}

    def validate_obj(section_name: str, value: Any, type_name: str):
        schema = defs.get(type_name)
        if not schema:
            return
        try:
            Draft7Validator(with_defs(schema), format_checker=fmt).validate(value)
        except Exception as e:
            errors.append(f"{section_name} validation error: {e}")

    # Singles
    if data.get("Company"):
        validate_obj("Company", data["Company"], "Company")
    if data.get("Websites"):
        validate_obj("Websites", data["Websites"], "Websites")
    if data.get("GoogleBusinessProfile"):
        validate_obj("GoogleBusinessProfile", data["GoogleBusinessProfile"], "GoogleBusinessProfile")
    if data.get("KnowledgePanel"):
        validate_obj("KnowledgePanel", data["KnowledgePanel"], "KnowledgePanel")
    if data.get("BacklinkMetrics"):
        validate_obj("BacklinkMetrics", data["BacklinkMetrics"], "BacklinkMetrics")
    if data.get("TechHealthReport"):
        validate_obj("TechHealthReport", data["TechHealthReport"], "TechHealthReport")
    if data.get("AssessmentScores"):
        validate_obj("AssessmentScores", data["AssessmentScores"], "AssessmentScores")

    # Arrays
    array_map = {
        "SocialProfiles": "SocialProfile",
        "Products": "Product",
        "Services": "Service",
        "Rankings": "RankingEntry",
        "ReviewAggregates": "ReviewAggregate",
        "JobListings": "JobListing",
        "EmployerRatings": "EmployerRating",
        "SentimentSamples": "SentimentSample",
        "MediaMentions": "MediaMention",
        "Events": "Event",
        "MediaAppearances": "MediaAppearance",
        "FundingRounds": "FundingRound",
        "OpenSourceRepos": "OpenSourceRepo",
        "Chatbots": "Chatbot",
        "Patents": "Patent",
        "Publications": "Publication",
        "Mentions": "Mention",
        # Websites.schema_org is validated implicitly if present in Websites
    }
    for key, type_name in array_map.items():
        arr = data.get(key, [])
        if not isinstance(arr, list):
            continue
        for i, item in enumerate(arr):
            try:
                schema = defs.get(type_name)
                if schema:
                    Draft7Validator(with_defs(schema), format_checker=fmt).validate(item)
            except Exception as e:
                errors.append(f"{key}[{i}] validation error: {e}")

    # Provenance
    prov = data.get("Provenance", [])
    if isinstance(prov, list):
        schema = defs.get("ProvenanceEntry")
        if schema:
            for i, p in enumerate(prov):
                try:
                    Draft7Validator(with_defs(schema), format_checker=fmt).validate(p)
                except Exception as e:
                    errors.append(f"Provenance[{i}] validation error: {e}")

    return (len(errors) == 0), errors


def choose_gemini_model() -> Optional[Any]:
    # Ensure env vars are loaded if possible
    _load_env_vars()
    if genai is None:
        print("Gemini package not installed. Install google-generativeai to enable Q&A.")
        return None
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set. Set it to enable Q&A.")
        return None
    genai.configure(api_key=api_key)
    # Try to list models and pick one with generateContent
    try:
        models = list(genai.list_models())
        supported: List[str] = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods or "generate_content" in methods:
                name = getattr(m, "name", "")
                if name.startswith("models/"):
                    name = name.split("/", 1)[1]
                supported.append(name)
        prefs = [
            os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b-latest",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-002",
            "gemini-1.5-pro",
        ]
        candidates: List[str] = []
        seen = set()
        for p in prefs:
            if p in supported and p not in seen:
                candidates.append(p)
                seen.add(p)
        for s in supported:
            if s not in seen:
                candidates.append(s)
                seen.add(s)
    except Exception:
        candidates = [
            os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-002",
            "gemini-1.5-flash-8b-latest",
            "gemini-1.5-pro-latest",
        ]
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            model = genai.GenerativeModel(name, system_instruction=SYSTEM_QA)
            model.generate_content("ping")
            return model
        except Exception as e:
            last_err = e
            continue
    print(f"Could not initialize a Gemini model. Last error: {last_err}")
    return None


def answer_questions(model: Any, data: Dict[str, Any], questions: List[str], log_path: Optional[str] = None) -> None:
    doc = json.dumps(data, ensure_ascii=False, indent=2)
    def append_log(entry: Dict[str, Any]):
        if not log_path:
            return
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
    for q in questions:
        prompt = (
            "You are given a JSON document representing a company's profile. "
            "Answer ONLY from this JSON. If unknown, say so.\n\n"
            f"[JSON]\n{doc}\n\n[QUESTION]\n{q}\n\n[ANSWER]"
        )
        try:
            resp = model.generate_content(prompt)
            ans = (resp.text or "").strip() or "(no answer)"
        except Exception as e:
            ans = f"(error: {e})"
        print(f"Q: {q}\nA: {ans}\n")
        append_log({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question": q,
            "answer": ans,
        })


def main():
    parser = argparse.ArgumentParser(description="Single-file Agent: crawl domain, save JSON, and Q&A via Gemini")
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--domain", help="Domain like example.com")
    g.add_argument("--email", help="Email like user@example.com")
    parser.add_argument("--ask", action="append", help="Ask a question after crawl (can repeat). Skips REPL if provided.")
    parser.add_argument("--out", help="Output JSON path (defaults to out/<domain>_agent.json)")
    parser.add_argument("--stdout", action="store_true", help="Print JSON to stdout (and optionally also save if --out is set)")
    parser.add_argument("--save-chat", nargs="?", const="__AUTO__", help="Save chat Q&A to a JSONL file. If no path is provided, a default path is used.")
    args = parser.parse_args()

    # Greet first (quiet when --stdout to keep output clean)
    if not args.stdout:
        print("Hello! I’m your crawl agent. I’ll fetch a domain’s website, save a structured JSON, and then answer your questions about it.\n")

    # Resolve target
    if args.email:
        domain = extract_domain(args.email)
        if not args.stdout:
            print(f"Using email provided. Target domain: {domain}")
    elif args.domain:
        domain = extract_domain(args.domain)
        if not args.stdout:
            print(f"Using domain provided: {domain}")
    else:
        prompt = "What domain should I find info on? (You can also paste an email): "
        target = input(prompt).strip()
        domain = extract_domain(target)

    base_url = resolve_base_url(domain)
    if not args.stdout:
        print(f"\nGreat — crawling {base_url} now. This may take a moment…")
    if args.stdout:
        # Suppress crawler library logs when producing stdout JSON
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            html = fetch_html(base_url)
    else:
        html = fetch_html(base_url)
    data = build_json(domain, base_url, html)

    # Validate against component schemas
    defs = load_schema_defs()
    ok, errs = validate_output(data, defs)
    if not args.stdout and not ok:
        print("Validation warnings (some fields may not meet schema exactly):")
        for e in errs[:10]:
            print(" -", e)
        if len(errs) > 10:
            print(f" - ... and {len(errs) - 10} more")

    # Save or print JSON
    out_path = args.out or f"out/{domain}_agent.json"
    if args.stdout:
        # Silence any accidental prints when piping
        with contextlib.redirect_stdout(sys.stdout), contextlib.redirect_stderr(io.StringIO()):
            print(json.dumps(data, ensure_ascii=False, indent=2))
    if args.out or not args.stdout:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if not args.stdout:
            print(f"Saved JSON to: {out_path}")

    # Q&A
    if args.ask:
        model = choose_gemini_model()
        if model:
            # Set up optional log path
            chat_log_path: Optional[str] = None
            if args.save_chat:
                if args.save_chat == "__AUTO__":
                    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
                    base = os.path.splitext(os.path.basename(out_path))[0]
                    chat_log_path = os.path.join("out", f"{base}_chat_{ts}.jsonl")
                else:
                    chat_log_path = args.save_chat
            answer_questions(model, data, args.ask, log_path=chat_log_path)
        return

    # Interactive chat
    if not args.stdout:
        print("\nAll set. Ask me anything about this company. Type 'exit' to quit. Type 'json' to print the file path.\n")
    model = choose_gemini_model()
    if not model:
        print("Q&A unavailable (Gemini not configured). Exiting.")
        return
    # Determine chat log path if requested
    chat_log_path: Optional[str] = None
    if args.save_chat:
        if args.save_chat == "__AUTO__":
            ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            base = os.path.splitext(os.path.basename(out_path))[0]
            chat_log_path = os.path.join("out", f"{base}_chat_{ts}.jsonl")
        else:
            chat_log_path = args.save_chat

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if q.lower() in {"json", "file", "path"}:
            print(out_path)
            continue
        answer_questions(model, data, [q], log_path=chat_log_path)


if __name__ == "__main__":
    main()
