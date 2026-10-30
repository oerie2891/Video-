"""Sorting, site identification, colours and page parsing. No network."""

from __future__ import annotations

import json

import pytest

from tubedex import htmlmeta
from tubedex.sites import (
    KNOWN_SITES,
    distance,
    generate_color,
    identify,
    normalize_hex,
    registrable_domain,
    text_color,
)
from tubedex.sorting import LETTER, NUMBER, SYMBOL, group_of, sort_titles

# --------------------------------------------------------------------------
# Ordering rule
# --------------------------------------------------------------------------


def test_symbols_then_numbers_then_letters():
    titles = ["zebra", "Apple", "10 things", "#1 hit", "2 girls", "!wow", "0 to 60", "émilie",
              "Banana", "(live)", "100 days"]
    assert sort_titles(titles) == [
        "!wow", "#1 hit", "(live)",          # symbols first
        "0 to 60", "2 girls", "10 things", "100 days",  # numbers by value, not text
        "Apple", "Banana", "émilie", "zebra",  # letters, case and accents ignored
    ]


def test_numbers_inside_titles_compare_by_value():
    assert sort_titles(["Part 10", "Part 2", "Part 1"]) == ["Part 1", "Part 2", "Part 10"]


def test_groups():
    assert group_of("  #tag") == SYMBOL
    assert group_of("3some") == NUMBER
    assert group_of("Ébène") == LETTER


def test_sort_is_total_and_stable():
    titles = ["abc", "ABC", "Abc"]
    assert sort_titles(titles) == sort_titles(list(reversed(titles)))


# --------------------------------------------------------------------------
# Sites and colours
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "key"),
    [
        ("https://www.pornhub.com/view_video.php?viewkey=abc", "pornhub"),
        ("https://de.pornhub.org/view_video.php?viewkey=abc", "pornhub"),
        ("https://www.xvideos2.com/video123/x", "xvideos"),
        ("https://www.xnxx.com/video-abc/x", "xnxx"),
        ("https://www.redtube.com/123", "redtube"),
        ("https://x.com/user/status/1", "x"),
        ("https://twitter.com/user/status/1", "x"),
        ("https://www.eporner.com/video-abc/x/", "eporner"),
        ("https://www.youporn.com/watch/1/x/", "youporn"),
        ("https://xhamster19.com/videos/x-123", "xhamster"),
        ("https://xhamster.desi/videos/x-123", "xhamster"),
        ("https://spankbang.com/abc/video/x", "spankbang.com"),
        ("https://www.example.co.uk/a", "example.co.uk"),
    ],
)
def test_identify(url, key):
    assert identify(url).key == key


def test_lookalike_domains_are_not_known_sites():
    assert identify("https://pornhubclone.com/v/1").key == "pornhubclone.com"
    assert identify("https://pornhub.evil.example/v/1").key == "evil.example"
    assert identify("https://xx.com/a").key == "xx.com"


def test_registrable_domain():
    assert registrable_domain("a.b.example.com") == "example.com"
    assert registrable_domain("example.com.au") == "example.com.au"


def test_fixed_colours():
    colours = {s.key: s.color for s in KNOWN_SITES}
    assert colours == {
        "pornhub": "#FF9000",
        "xvideos": "#FF0000",
        "xnxx": "#0066FF",
        "redtube": "#8A0303",
        "x": "#000000",
        "eporner": "#36454F",
        "youporn": "#FF69B4",
        "xhamster": "#800080",
    }


def test_generated_colours_stay_distinct():
    palette = [s.color for s in KNOWN_SITES]
    generated = []
    for _ in range(40):
        colour = generate_color(palette + generated)
        assert colour not in palette + generated
        generated.append(colour)
    # The first dozen extra sites are all clearly distinguishable...
    first = palette + generated[:12]
    gaps = [distance(a, b) for i, a in enumerate(first) for b in first[i + 1:]]
    assert min(gaps) > 20
    # ...and even at 48 sites no two tags are identical.
    assert len(set(generated)) == 40


def test_generation_is_deterministic():
    used = ["#FF9000", "#000000"]
    assert generate_color(used) == generate_color(list(used))


def test_text_colour_and_hex():
    assert text_color("#000000") == "#FFFFFF"
    assert text_color("#FF9000") == "#000000"
    assert normalize_hex("ff9000") == "#FF9000"
    with pytest.raises(ValueError):
        normalize_hex("orange")


# --------------------------------------------------------------------------
# Page parsing
# --------------------------------------------------------------------------

VIDEO_PAGE = """<!doctype html><html><head>
<title>Fallback title - Some Tube</title>
<meta property="og:title" content="Sunset &amp; Waves">
<meta property="og:type" content="video.movie">
<meta property="og:image" content="https://cdn.example/t.jpg">
<meta property="video:tag" content="beach"><meta property="video:tag" content="outdoor">
<meta property="video:actor" content="Meta Person">
<script type="application/ld+json">
{"@context": "https://schema.org", "@graph": [
  {"@type": "ImageObject", "url": "https://cdn.example/logo.png"},
  {"@type": "VideoObject", "name": "Sunset & Waves", "uploadDate": "2024-03-05T10:00:00+00:00",
   "duration": "PT12M30S", "author": {"@type": "Person", "name": "Studio One"},
   "actor": [{"@type": "Person", "name": "Ava Example"}, {"@type": "Person", "name": "Bo Sample"}],
   "genre": ["Outdoor", "Romantic"], "keywords": "beach, sunset",
   "interactionStatistic": [
     {"@type": "InteractionCounter", "interactionType": {"@type": "WatchAction"}, "userInteractionCount": 1234567},
     {"@type": "InteractionCounter", "interactionType": "http://schema.org/LikeAction", "userInteractionCount": "8,910"},
     {"@type": "InteractionCounter", "interactionType": "https://schema.org/CommentAction", "userInteractionCount": 42}
   ]}
]}
</script></head><body>
<nav><a href="/pornstars/popular-person">Popular Person</a></nav>
<div class="video-metadata">
  <ul>
    <li class="main-uploader"><a href="/channels/studio-one"><span class="name">Studio One</span><span class="count">12k</span></a></li>
    <li><a class="is-pornstar" href="/pornstars/ava"><span class="name">Ava Example</span></a></li>
    <li><a class="is-keyword" href="/tags/beach">beach</a></li>
  </ul>
</div>
<div id="v-views"><strong class="mobile-hide">1.2M</strong> views</div>
<span class="rating-good-nbr">8.9k</span><span class="rating-bad-nbr">310</span>
</body></html>"""


def test_json_ld_layer():
    data = htmlmeta.from_json_ld(htmlmeta.parse(VIDEO_PAGE))
    assert data["kind"] == "video"  # VideoObject preferred over the logo ImageObject
    assert data["title"] == "Sunset & Waves"
    assert data["published"] == "2024-03-05"
    assert data["duration"] == 750
    assert data["channel"] == "Studio One"
    assert data["people"] == ["Ava Example", "Bo Sample"]
    assert data["categories"] == ["Outdoor", "Romantic"]
    assert data["tags"] == ["beach", "sunset"]
    assert (data["views"], data["likes"], data["comments"]) == (1234567, 8910, 42)


def test_meta_tag_layer():
    data = htmlmeta.from_meta_tags(htmlmeta.parse(VIDEO_PAGE))
    assert data["title"] == "Sunset & Waves"
    assert data["kind"] == "video"
    assert data["tags"] == ["beach", "outdoor"]
    assert data["people"] == ["Meta Person"]
    assert data["thumbnail"] == "https://cdn.example/t.jpg"


def test_markup_layer_ignores_navigation_links():
    page = htmlmeta.parse(VIDEO_PAGE)
    data = htmlmeta.from_markup(page, VIDEO_PAGE)
    assert data["people"] == ["Ava Example"]  # not "Popular Person" from the nav
    assert data["channel"] == "Studio One"    # the name span, not "Studio One12k"
    assert data["tags"] == ["beach"]
    assert (data["views"], data["likes"], data["dislikes"]) == (1_200_000, 8900, 310)


def test_malformed_json_ld_and_html_do_not_crash():
    page = htmlmeta.parse('<script type="application/ld+json">{"@type": "VideoObject", '
                          '"name": "X",}</script><div><a class="is-pornstar">A<b>B</div>')
    assert htmlmeta.from_json_ld(page)["title"] == "X"


@pytest.mark.parametrize(
    ("raw", "value"),
    [(1234, 1234), ("1,234", 1234), ("1.2K", 1200), ("3.4M views", 3_400_000),
     ("12 345", 12345), ("UserPlays:5678", 5678), ("1.234.567", 1234567), ("", None),
     ("n/a", None), (None, None), (True, None)],
)
def test_parse_count(raw, value):
    assert htmlmeta.parse_count(raw) == value


def test_parse_duration_and_date():
    assert htmlmeta.parse_duration("PT1H2M3S") == 3723
    assert htmlmeta.parse_duration("615") == 615
    assert htmlmeta.parse_duration("soon") is None
    assert htmlmeta.parse_date("20240305") == "2024-03-05"
    assert htmlmeta.parse_date("2024-13-01") is None


def test_magazine_json_ld():
    html = ('<script type="application/ld+json">' + json.dumps({
        "@type": "PublicationIssue", "name": "Issue 12", "datePublished": "1999-07-01",
        "publisher": {"@type": "Organization", "name": "Glossy Press"},
        "genre": "Glamour"}) + "</script>")
    data = htmlmeta.from_json_ld(htmlmeta.parse(html))
    assert data["kind"] == "magazine"
    assert data["channel"] == "Glossy Press"
    assert data["categories"] == ["Glamour"]
