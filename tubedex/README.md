# tubedex

A private library for adult video, photo, GIF and magazine links. Paste a link
and tubedex records the title, kind, uploader/channel, the people tagged, views,
likes, dislikes, comments, publish date, duration, best quality, categories,
tags and the website it came from. Every category becomes a playlist, and every
website gets its own coloured tag.

It runs on your own computer. The library is one SQLite file. There's a web UI
and a CLI.

```bash
tubedex add "https://www.pornhub.com/view_video.php?viewkey=…"   # fetch + print stats
tubedex add --file links.txt                                     # one link per line
tubedex serve                                                    # web UI: http://127.0.0.1:8765
tubedex playlists                                                # categories A–Z
tubedex export ~/Playlists                                       # a real folder per category
```

## Install

You need Python 3.10 or newer.

```bash
cd tubedex
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
```

Update yt-dlp often (`pip install -U yt-dlp`). Sites change their pages all
the time, and yt-dlp usually ships a fix within days.

## Feature 1: statistics for every link

tubedex reads each link in layers. It uses the first layer that has a value for
a field and never mixes values from different layers:

1. **yt-dlp**: maintained extractors for every site listed below.
2. **JSON-LD**: the structured data that sites publish for search engines.
   This often holds view, like and comment counts and the cast.
3. **Meta tags**: OpenGraph `og:title`, `video:tag` and `video:actor`.
4. **Markup hints**: class names that some tube sites put on performer, tag and
   uploader links.

The page itself is only downloaded if yt-dlp left fields empty. `tubedex show
ID` or the detail view lists where each value came from, and `tubedex inspect
URL` runs every layer without saving anything.

**How much each site returns varies.** This table lists what yt-dlp's own
extractor provides. Anything missing has to come from layers 2–4 or from you:

| Site | From yt-dlp | People tagged |
|---|---|---|
| Pornhub | uploader, categories, tags, views, likes, dislikes, comments, date | ✓ |
| xHamster | uploader, categories, views, likes, dislikes, comments, date | page only |
| YouPorn | uploader, categories, tags, views, comments, date | page only |
| RedTube | views, date | page only |
| Eporner | views, date | page only |
| XVideos / XNXX | title, duration only (views on XNXX) | page only |
| X (Twitter) | uploader, likes, comments, views, date: **needs cookies** | – |
| other sites | yt-dlp generic extractor, then the page | page only |

"Page only" means JSON-LD, meta tags or markup hints, if the site exposes them.
Any field can be filled in or corrected by hand. Hand edits are locked, so a
later refresh won't overwrite them:

```bash
tubedex edit 12 --people "Name One, Name Two" --categories "Amateur, Outdoor"
tubedex edit 12 --unlock      # let refreshes overwrite again
```

**Kinds.** Kinds are video, photo, GIF, magazine and link. tubedex works out the
kind from the file extension or content type, URL paths such as `/gif/` and
`/album/`, and the page's structured data. Override it with `--kind` or in the
UI. Magazines rarely publish views or likes. For a magazine, the publisher goes
in *Channel* and the models go in *People*.

**Stats over time.** `tubedex refresh --all` re-reads every item. Each fetch is
kept, so `show` reports views gained since you added the item. Views per day
and the like ratio are calculated too. A refresh that fails, for example
because a video was deleted or geo-blocked, keeps the data you already had.

**Duplicates.** The same video under two URLs, such as a mirror domain or a
tracking parameter, is detected and stored only once.

### Sign-in, age gates and geo-blocks

- Pornhub and some of its sister sites block whole countries and US states that
  have age-verification laws. A fetch from those places fails. That's the site's
  decision, not a bug in tubedex.
- X needs a signed-in session for most adult posts. Pass your browser's
  cookies:
  ```bash
  tubedex add URL --cookies-from-browser firefox      # or chrome, edge, safari, brave…
  tubedex serve --cookies-from-browser chrome
  ```
  You can also use `--cookies cookies.txt`, or set `TUBEDEX_COOKIES` /
  `TUBEDEX_COOKIES_FROM_BROWSER`.

## Feature 2: playlists and coloured site tags

**One playlist per category.** tubedex treats different spellings of a category
across sites as one: `Big-Tits`, `big tits` and `BIG_TITS` all go into one
playlist. The playlist takes the most common spelling as its name. Items with
no category go into *Uncategorised*, which always sits at the end of the list.

**Ordering rule.** Playlists and the titles inside them follow the same rule:

1. Symbols first (`!`, `#`, `(` …)
2. Then numbers, counted up by value: `0`, `2`, `10`, `100`. Plain text order
   would put `10` before `2`.
3. Then letters A–Z, ignoring case and accents, so `Émilie` sorts with E.

Numbers inside a title also compare by value, so `Part 2` comes before
`Part 10`.

**Site colours.**

| Site | Colour |
|---|---|
| Pornhub | `#FF9000` orange |
| XVideos | `#FF0000` red |
| XNXX | `#0066FF` blue |
| RedTube | `#8A0303` blood red |
| X | `#000000` black |
| Eporner | `#36454F` charcoal grey |
| YouPorn | `#FF69B4` pink |
| xHamster | `#800080` purple |

Every other site gets a colour the first time it appears. tubedex picks it from
a 360-colour pool, choosing the one that looks most different (CIELAB distance)
from every colour already in use. The first dozen or so extra sites get clearly
different colours. With dozens of sites, new colours are only slightly
different, but no two are ever the same. Colours are saved, so a site keeps its
colour. Mirror domains (`xvideos2.com`, `xhamster19.com`, `de.pornhub.com`)
share their main site's tag. Text on each tag is black or white, whichever is
easier to read.

You can change any colour, for example to match a site's exact orange:

```bash
tubedex sites                          # every site, its colour and item count
tubedex sites color pornhub '#FFA31A'
tubedex sites color pornhub reset
```

In the web UI, click a site's swatch to change its colour.

**Export to real folders.** `tubedex export DIR` creates one folder per category
and a shortcut file for each item. Shortcuts are `.url` on Windows, `.webloc`
on macOS and `.desktop` on Linux. Pick one with `--style`. Your file manager
sorts names with its own rules, so folder and file names are numbered (`001 `,
`002 `…) to keep tubedex's order. Use `--no-numbers` to turn that off. A file
name can't hold a colour, so each file name includes the site in brackets.
`DIR/index.html` shows the same playlists with coloured tags.

## Privacy

- The library (`~/.tubedex/library.db`, or `$TUBEDEX_DB` / `--db`) is a record
  of your viewing interests. Keep it somewhere only you can read. Full-disk
  encryption is the simple answer.
- The web UI listens on `127.0.0.1` only. Don't pass `--host 0.0.0.0` unless you
  want everyone on your network to see your library.
- Thumbnails load straight from the sites' image servers. The **Thumbnails:
  show / blur / hide** button in the header blurs or hides them, and your
  choice is remembered.

## Files

```
tubedex/
  extract.py   yt-dlp + page layers -> Record
  htmlmeta.py  JSON-LD, meta tags, markup hints, number/date parsing
  library.py   SQLite store, hand-edit locks, playlists, stat history
  sites.py     site identification, fixed colours, colour generation
  sorting.py   the symbols -> numbers -> A-Z rule
  export.py    folder export
  server.py    FastAPI app for the web UI
  cli.py       command line
  web/         single-page UI
```

Run the tests with `pip install -e '.[dev]' && pytest`. They use stand-in pages
and a stand-in yt-dlp, so they need no network.
