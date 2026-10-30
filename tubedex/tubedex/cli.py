"""Command-line interface.

    tubedex add URL [URL ...]          # fetch and store; prints the stats
    tubedex add --file links.txt       # one link per line
    tubedex list --category Amateur    # filter by category / person / site / kind
    tubedex show 12                    # every stat for one item
    tubedex edit 12 --people "A, B"    # fix or fill a field by hand
    tubedex refresh --all              # re-read views/likes/comments
    tubedex playlists                  # categories A-Z with counts
    tubedex sites                      # site tags and their colours
    tubedex sites color pornhub '#FFA31A'
    tubedex export ~/Playlists         # a real folder per category
    tubedex inspect URL                # what each source found, for debugging
    tubedex serve                      # web UI on http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .extract import KINDS, Extractor, FetchOptions, normalize_url
from .library import SORTS, Library, NotFound, default_path

# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------


def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def tag(site: dict[str, Any], color: bool | None = None) -> str:
    """The site's coloured tag, in 24-bit terminal colour when supported."""
    label = f" {site['name']} "
    if not (_use_color() if color is None else color):
        return f"[{site['name']}]"
    bg = site["color"].lstrip("#")
    fg = site["text_color"].lstrip("#")
    r, g, b = (int(bg[i:i + 2], 16) for i in (0, 2, 4))
    fr, fg_, fb = (int(fg[i:i + 2], 16) for i in (0, 2, 4))
    return f"\x1b[48;2;{r};{g};{b}m\x1b[38;2;{fr};{fg_};{fb}m\x1b[1m{label}\x1b[0m"


def _num(value: int | None) -> str:
    if value is None:
        return "–"
    return f"{value:,}"


def _duration(seconds: int | None) -> str:
    if not seconds:
        return "–"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _line(item: dict[str, Any]) -> str:
    title = item["title"] or item["url"]
    stats = f"{_num(item['views'])} views · {_num(item['likes'])} likes"
    status = "" if item["status"] in ("ok", "partial") else f"  ({item['status']})"
    return f"{item['id']:>5}  {tag(item['site'])} {item['kind']:<8} {title}  — {stats}{status}"


def print_item(item: dict[str, Any]) -> None:
    def row(label: str, value: Any) -> None:
        if value in (None, "", []):
            value = "–"
        elif isinstance(value, list):
            value = ", ".join(value)
        print(f"  {label:<13}{value}")

    print(f"#{item['id']}  {tag(item['site'])}  {item['title'] or '(no title)'}")
    row("Kind", item["kind"])
    row("Website", f"{item['site']['name']}  ({item['site']['color']})")
    row("Link", item["page_url"] or item["url"])
    row("Channel", item["channel"])
    row("People", item["people"])
    row("Categories", item["categories"])
    row("Tags", item["tags"])
    row("Views", _num(item["views"]))
    if item["views_per_day"] is not None:
        row("Views/day", f"{item['views_per_day']:,.1f}")
    row("Likes", _num(item["likes"]))
    if item["dislikes"] is not None:
        row("Dislikes", _num(item["dislikes"]))
    if item["rating"] is not None:
        row("Rating", f"{item['rating']}% liked")
    row("Comments", _num(item["comments"]))
    row("Published", item["published"])
    row("Duration", _duration(item["duration"]))
    if item["height"]:
        row("Best quality", f"{item['height']}p")
    history = item.get("history") or []
    if len(history) > 1 and history[0]["views"] is not None and history[-1]["views"] is not None:
        gained = history[-1]["views"] - history[0]["views"]
        row("Since added", f"{gained:+,} views over {len(history)} fetches")
    row("Status", item["status"] + (f" — {item['error']}" if item["error"] else ""))
    if item["locked"]:
        row("Hand-edited", item["locked"])
    for warning in item["warnings"]:
        print(f"  ! {warning}")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def _extractor(args: argparse.Namespace) -> Extractor:
    options = FetchOptions.from_env()
    if getattr(args, "cookies", None):
        options.cookies = args.cookies
    if getattr(args, "cookies_from_browser", None):
        options.cookies_from_browser = args.cookies_from_browser
    return Extractor(options)


def _read_links(args: argparse.Namespace) -> list[str]:
    links = list(args.urls)
    if args.file:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text("utf-8")
        links += [line.strip() for line in text.splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    return links


def cmd_add(lib: Library, args: argparse.Namespace) -> int:
    links = _read_links(args)
    if not links:
        print("no links given", file=sys.stderr)
        return 2
    fetch = _extractor(args)
    failures = 0
    for raw in links:
        try:
            item, created = lib.add(raw, args.kind)
        except ValueError as exc:
            print(f"skip {raw!r}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if not created and not args.refresh:
            print(f"already in library as #{item['id']}: {item['title'] or item['url']}")
            continue
        print(f"fetching {item['url']} …", file=sys.stderr)
        item = lib.fetch(item["id"], fetch)
        if item.get("duplicate_of"):
            print(f"same item as #{item['duplicate_of']} (different link) — not added twice")
            continue
        print_item(item)
        print()
        failures += item["status"] == "error"
    return 1 if failures else 0


def cmd_list(lib: Library, args: argparse.Namespace) -> int:
    items = lib.items(category=args.category, uncategorised=args.uncategorised,
                      person=args.person, tag=args.tag, site=args.site, kind=args.kind,
                      search=args.search, status=args.status, sort=args.sort)
    if args.json:
        print(json.dumps(items, indent=2, ensure_ascii=False))
        return 0
    for item in items:
        print(_line(item))
    print(f"{len(items)} item(s)", file=sys.stderr)
    return 0


def cmd_show(lib: Library, args: argparse.Namespace) -> int:
    for item_id in args.ids:
        item = lib.get(item_id)
        if args.json:
            print(json.dumps(item, indent=2, ensure_ascii=False))
        else:
            print_item(item)
            print()
    return 0


def cmd_edit(lib: Library, args: argparse.Namespace) -> int:
    if args.unlock:
        print_item(lib.unlock(args.id))
        return 0
    changes = {name: getattr(args, name) for name in (
        "title", "kind", "channel", "people", "categories", "tags", "views", "likes",
        "dislikes", "comments", "published", "duration",
    ) if getattr(args, name) is not None}
    if not changes:
        print("nothing to change; see `tubedex edit --help`", file=sys.stderr)
        return 2
    print_item(lib.edit(args.id, changes))
    return 0


def cmd_refresh(lib: Library, args: argparse.Namespace) -> int:
    if args.all:
        ids = [item["id"] for item in lib.items(sort="added")]
    elif args.failed:
        ids = [i["id"] for i in lib.items(status="error")] + \
              [i["id"] for i in lib.items(status="pending")]
    else:
        ids = args.ids
    if not ids:
        print("nothing to refresh: give ids, --all or --failed", file=sys.stderr)
        return 2
    fetch = _extractor(args)
    for item_id in ids:
        item = lib.fetch(item_id, fetch)
        print(_line(item))
    return 0


def cmd_rm(lib: Library, args: argparse.Namespace) -> int:
    for item_id in args.ids:
        lib.delete(item_id)
        print(f"deleted #{item_id}")
    return 0


def cmd_playlists(lib: Library, args: argparse.Namespace) -> int:
    lists = lib.playlists()
    if args.name:
        wanted = args.name.casefold()
        match = next((p for p in lists if p["name"].casefold() == wanted), None)
        if match is None:
            print(f"no playlist {args.name!r}", file=sys.stderr)
            return 1
        items = (lib.items(uncategorised=True) if match.get("uncategorised")
                 else lib.items(category=match["name"]))
        print(f"{match['name']} ({len(items)})")
        for item in items:
            print(_line(item))
        return 0
    if args.json:
        print(json.dumps(lists, indent=2, ensure_ascii=False))
        return 0
    for playlist in lists:
        print(f"{playlist['count']:>6}  {playlist['name']}")
    return 0


def cmd_sites(lib: Library, args: argparse.Namespace) -> int:
    if args.action == "color":
        if not args.key:
            print("usage: tubedex sites color SITE '#RRGGBB'  (or 'reset')", file=sys.stderr)
            return 2
        color = None if args.color in (None, "reset", "default") else args.color
        site = lib.set_site_color(args.key, color)
        print(f"{tag(site)}  {site['key']}  {site['color']}")
        return 0
    for site in lib.sites(used_only=not args.all):
        custom = "  (custom)" if site["custom"] else ""
        print(f"{tag(site)}  {site['key']:<20} {site['color']}  {site['count']:>5} item(s){custom}")
    return 0


def cmd_export(lib: Library, args: argparse.Namespace) -> int:
    from .export import export

    result = export(lib, args.out, style=args.style, numbered=not args.no_numbers)
    print(f"wrote {result['shortcuts']} shortcut(s) in {result['folders']} folder(s) "
          f"to {result['path']}")
    return 0


def cmd_inspect(lib: Library, args: argparse.Namespace) -> int:
    record = _extractor(args).fetch(normalize_url(args.url), args.kind)
    data = record.to_dict()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0 if not record.error else 1


def cmd_serve(lib: Library, args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    if args.cookies:
        os.environ["TUBEDEX_COOKIES"] = args.cookies
    if args.cookies_from_browser:
        os.environ["TUBEDEX_COOKIES_FROM_BROWSER"] = args.cookies_from_browser
    print(f"tubedex on http://{args.host}:{args.port}  (library: {lib.path})", file=sys.stderr)
    uvicorn.run(create_app(lib), host=args.host, port=args.port, log_level="warning")
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def _cookie_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cookies", metavar="FILE",
                   help="Netscape cookies.txt for sites that need you signed in")
    p.add_argument("--cookies-from-browser", metavar="BROWSER[:PROFILE]",
                   help="read cookies from chrome, firefox, edge, safari, brave, …")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tubedex", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"tubedex {__version__}")
    parser.add_argument("--db", metavar="PATH",
                        help=f"library file (default: $TUBEDEX_DB or {default_path()})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="add links and fetch their stats")
    p.add_argument("urls", nargs="*")
    p.add_argument("--file", "-f", help="text file with one link per line ('-' for stdin)")
    p.add_argument("--kind", choices=KINDS, help="force the kind instead of detecting it")
    p.add_argument("--refresh", action="store_true", help="re-fetch links already in the library")
    _cookie_args(p)
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("list", help="list items")
    p.add_argument("--category", "-c")
    p.add_argument("--uncategorised", action="store_true")
    p.add_argument("--person", "-p")
    p.add_argument("--tag", "-t")
    p.add_argument("--site", "-s", help="site key, e.g. pornhub or example.com")
    p.add_argument("--kind", "-k", choices=KINDS)
    p.add_argument("--status", choices=("pending", "ok", "partial", "error"))
    p.add_argument("--search", "-q")
    p.add_argument("--sort", choices=SORTS, default="title")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="every stat for one or more items")
    p.add_argument("ids", nargs="+", type=int)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("edit", help="set fields by hand (kept through refreshes)")
    p.add_argument("id", type=int)
    p.add_argument("--title")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--channel")
    p.add_argument("--people", help="comma-separated")
    p.add_argument("--categories", help="comma-separated")
    p.add_argument("--tags", help="comma-separated")
    for name in ("views", "likes", "dislikes", "comments", "duration"):
        p.add_argument(f"--{name}", type=int)
    p.add_argument("--published", help="YYYY-MM-DD")
    p.add_argument("--unlock", action="store_true",
                   help="forget hand edits so the next refresh overwrites them")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("refresh", help="re-fetch stats")
    p.add_argument("ids", nargs="*", type=int)
    p.add_argument("--all", action="store_true")
    p.add_argument("--failed", action="store_true", help="only items that errored or are pending")
    _cookie_args(p)
    p.set_defaults(func=cmd_refresh)

    p = sub.add_parser("rm", help="delete items")
    p.add_argument("ids", nargs="+", type=int)
    p.set_defaults(func=cmd_rm)

    p = sub.add_parser("playlists", help="categories A-Z, or the items in one")
    p.add_argument("name", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_playlists)

    p = sub.add_parser("sites", help="site tags and colours")
    p.add_argument("action", nargs="?", choices=("list", "color"), default="list")
    p.add_argument("key", nargs="?")
    p.add_argument("color", nargs="?", help="#RRGGBB, or 'reset'")
    p.add_argument("--all", action="store_true", help="include sites with no items")
    p.set_defaults(func=cmd_sites)

    p = sub.add_parser("export", help="write a folder per category of link shortcuts")
    p.add_argument("out")
    p.add_argument("--style", choices=("url", "webloc", "desktop"),
                   help="shortcut format (default: this OS's native one)")
    p.add_argument("--no-numbers", action="store_true",
                   help="don't number names (your file manager will pick the order)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("inspect", help="fetch a link without saving; show every source")
    p.add_argument("url")
    p.add_argument("--kind", choices=KINDS)
    _cookie_args(p)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("serve", help="run the web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    _cookie_args(p)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lib = Library(args.db)
    try:
        return args.func(lib, args)
    except NotFound as exc:
        print(f"not found: {exc.args[0]}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        lib.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
