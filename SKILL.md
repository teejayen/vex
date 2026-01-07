---
name: vex
description: Extract structured, searchable content from screenshots using vision capabilities. Use when the user wants to process screenshots, extract text from images, categorise visual content, or make their screenshot collection searchable. You will analyse images directly and output structured JSON.
license: MIT
---

# vex - Vision Extraction for Screenshots

Analyse screenshots and extract structured, searchable content.

## When to Use

- User has screenshots they want to make searchable
- User wants to extract text, recipes, code, or other content from images
- User needs to categorise and organise screenshots
- User mentions having unorganised screenshots

## How to Process Screenshots

### 1. Find Images

Use Glob to find supported image files:
```
**/*.png, **/*.jpg, **/*.jpeg, **/*.gif, **/*.webp
```

### 2. Analyse Each Image

Read each image file and analyse it. Determine the content type and extract accordingly:

**RECIPE**: Title, full ingredients with quantities, complete method steps, servings, cook time, source.

**CODE/TECHNICAL**: Language/technology, what it does, extract visible code verbatim, any commands shown.

**ERROR MESSAGE**: Exact error text, platform/app, error codes, stack trace, likely cause if obvious.

**ARTICLE/TEXT**: Key points summary, notable quotes, author, source, date if visible.

**CHAT/SOCIAL**: Who said what, platform, context, any links or key info shared.

**DOCUMENT**: Document type, key content, dates, names, reference numbers, important details.

**MEME/IMAGE**: Describe the image, transcribe all text overlays, explain the joke/context if relevant.

**UI/APP**: App name, what screen/feature is shown, any settings or data displayed.

**SHOPPING/PRODUCT**: Item name, price, store/site, any specs or details, link if visible.

**MAP/LOCATION**: Place name, full address, coordinates if shown, directions, context for why saved.

**RECEIPT/TRANSACTION**: Merchant, amount, date, order/confirmation number, items if listed.

**BOOKING/EVENT**: Event name, date, time, venue/location, confirmation number, ticket details.

**CONTACT INFO**: Name, phone, email, company, role, address - extract all contact details.

**MUSIC/MEDIA**: Song title, artist, album, playlist name, platform.

**QUOTE/INSPIRATION**: The exact quote, attribution/author, source.

**HEALTH/FITNESS**: Metrics shown, values, dates, app name, what's being tracked.

**SETTINGS/CONFIG**: App name, what settings are shown, current values, why this might matter.

### 3. Output Format

For each image, output a JSON object:

```json
{
    "path": "/path/to/image.png",
    "type": "recipe|code|error|article|chat|document|meme|ui|shopping|map|receipt|booking|contact|music|quote|health|settings|other",
    "category": "food|tech|work|social|finance|health|travel|entertainment|reference|personal|shopping|other",
    "summary": "1-2 sentence description of what this screenshot contains",
    "extracted_content": "The main extracted content - be thorough, this is the primary value",
    "tags": ["up to 5 relevant tags for searching"],
    "source": "source website/app if identifiable, null otherwise"
}
```

### 4. Save Results

Append each result as a line to a JSONL file (one JSON object per line).

## Processing Guidelines

- **Be thorough** with extracted_content - for recipes include everything to cook it, for code include the actual code, for receipts include all transaction details
- **Identify the source** when visible (website URL, app name, etc.)
- **Choose tags** that will help with searching later
- **Handle errors gracefully** - if an image can't be processed, note the error and continue

## Organising Results

After processing, files can be sorted into category folders:
- Create a folder for each category (food/, tech/, work/, etc.)
- Move or copy files based on their extracted category

## Example Output

```json
{"path": "/screenshots/IMG_001.png", "type": "article", "category": "tech", "summary": "Blog post about building Arc, a thinking partner that remembers context between sessions", "extracted_content": "Building Arc: A Thinking Partner That Remembers\n\nArc is a personal AI system built on Claude that maintains context across sessions. Key features:\n- Persistent state via markdown files\n- Journal for pattern detection over time\n- Decision capture with reasoning\n- Weekly reviews surfacing insights\n\nThe difference between a stateless chatbot and a genuine thinking partner is memory.", "tags": ["ai", "claude", "productivity", "arc", "thinking-partner"], "source": "tim.neilen.com.au"}
{"path": "/screenshots/IMG_002.png", "type": "quote", "category": "reference", "summary": "Quote about being present from Tim Neilen's quotes collection", "extracted_content": "Being is enough.\n\nYou don't need to constantly produce, achieve, or prove yourself. Sometimes the most profound thing you can do is simply exist fully in the moment.", "tags": ["mindfulness", "presence", "philosophy", "quotes"], "source": "tim.neilen.com.au"}
```
