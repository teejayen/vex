# vex

I have almost 12,000 screenshots. Phone, laptop, years of them.

The pattern is always the same: see something interesting, screenshot it, think "I'll come back to this." Never do.

Recipes I meant to cook. Error messages I needed to debug. Articles I wanted to read properly. Code snippets. Memes. Random thoughts someone posted that resonated.

All sitting in folders, unsearchable, effectively lost.

This is a perfect task to outsource to AI. Each image takes a human 10-30 seconds to process mentally. A vision model does it in under a second, for fractions of a cent.

**vex** (vision extraction) batch-processes screenshots through Claude. It looks at each image, figures out what it is, and extracts the useful information. A recipe becomes searchable text. Technical content gets categorised. Pure images get described.

The output is a JSON dump of everything - searchable, sortable, finally usable.

## The Problem It Solves

Screenshots are write-only storage. Easy to capture, impossible to retrieve.

Search doesn't work on images. You can't grep a photo of a recipe. You can't find that error message from six months ago unless you remember exactly when you saw it.

This tool turns screenshots into structured data. Now they're searchable.

## Setup

```bash
git clone https://github.com/teejayen/vex.git
cd vex
uv sync

# Add your API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

## Usage

```bash
# Test with a few images first
vex process /path/to/screenshots --limit 10

# Process a directory (realtime, one at a time)
vex process /path/to/screenshots

# Use batch API for large sets (50% cheaper, results in <24hr)
vex process /path/to/screenshots --batch

# Check batch progress
vex status msgbatch_xxx

# Download batch results
vex results msgbatch_xxx

# Sort processed files into category folders
vex organise results.jsonl --target /path/to/sorted/
```

## What It Extracts

The extraction adapts to what it sees:

| Type | What You Get |
|------|--------------|
| Recipe | Title, full ingredients, method steps, servings, cook time |
| Code/Technical | Language, explanation, the actual code verbatim |
| Error Message | Exact error text, platform, stack trace, likely cause |
| Article/Text | Key points, quotes, author, source |
| Chat/Social | Who said what, platform, context, links shared |
| Document | Type, key content, dates, reference numbers |
| Meme/Image | Description, all text transcribed, context |
| UI/App | App name, screen shown, settings or data displayed |
| Shopping/Product | Item, price, store, specs, link |
| Map/Location | Place name, full address, directions |
| Receipt/Transaction | Merchant, amount, date, order number, items |
| Booking/Event | Event, date, time, venue, confirmation number |
| Contact Info | Name, phone, email, company, address |
| Music/Media | Song, artist, album, playlist, platform |
| Quote/Inspiration | The quote, attribution, source |
| Health/Fitness | Metrics, values, dates, what's tracked |
| Settings/Config | App, what settings, current values |

Everything gets a category and tags for filtering later.

## Output

Results land in a JSONL file (one JSON object per line):

```json
{
    "path": "/screenshots/IMG_4521.png",
    "type": "recipe",
    "category": "food",
    "summary": "Thai green curry recipe from Bon Appetit",
    "extracted_content": "Ingredients: 400ml coconut milk, 2 tbsp green curry paste...",
    "tags": ["thai", "curry", "dinner", "bon-appetit"],
    "source": "bonappetit.com",
    "processed_at": "2025-01-07T12:00:00",
    "tokens_used": 1847
}
```

## Costs

Using Haiku 4.5 for 12,000 screenshots: roughly $10-15 USD.

Batch mode cuts that in half.

| Model | Realtime | Batch Mode |
|-------|----------|------------|
| Haiku 4.5 | ~$12 | ~$6 |
| Sonnet 4.5 | ~$36 | ~$18 |

## Commands

```
vex process <directory>     Extract content from images
vex status <batch_id>       Check batch processing status
vex results <batch_id>      Download completed batch results
vex organise <jsonl>        Sort files into category folders
```

### Process Options

```
-o, --output FILE     Output file (default: vex-results.jsonl)
-m, --model MODEL     Model to use (default: claude-haiku-4-5-20251001)
-r, --rate SECONDS    Delay between requests (default: 0.5)
-l, --limit N         Process only N images
--batch               Use batch API (50% off, <24hr processing)
--no-resume           Start fresh, ignore previous progress
--dry-run             List files without processing
```

### Organise Options

```
--target DIR          Target directory for sorted files (required)
--copy                Copy files instead of moving them
```

## Resume Support

Processing saves progress as it goes. If it stops, run the same command again - it picks up where it left off.

For batch mode, use `vex status` and `vex results` to check progress and download when ready.

## What Next?

The extracted data becomes useful beyond search. Feed it to a personal knowledge system, surface patterns in what you capture, auto-route content to the right places.

I built [Arc](https://tim.neilen.com.au/2025/12/29/building-arc-a-thinking-partner-that-remembers/) - a thinking partner that holds context between sessions. Screenshot data is perfect input: attention signals, things that mattered enough to capture, patterns over time.

---

Built by [Tim Neilen](https://tim.neilen.com.au) because I was tired of screenshots being a black hole.

Written with AI. I provided the problem and direction; Claude wrote the code. [More on how I use AI](https://tim.neilen.com.au/ai).

MIT License.
