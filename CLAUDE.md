# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Jekyll-based GitHub Pages blog using the Chirpy theme (v7.3+). The blog focuses on AI/ML research, particularly efficient ML, on-device AI, and Vision-Language-Action models. Content is written in Korean.

## Common Commands

### Development Server
```bash
# Run local development server with live reload
bash tools/run.sh

# Run with custom host
bash tools/run.sh -H 0.0.0.0

# Run in production mode
bash tools/run.sh -p
```

### Build and Test
```bash
# Build site and run html-proofer tests
bash tools/test.sh

# Build for production (used by CI/CD)
JEKYLL_ENV=production bundle exec jekyll b -d "_site"
```

### Dependency Management
```bash
# Install dependencies
bundle install

# Update dependencies
bundle update
```

## Content Structure

### Blog Posts
- **Published posts**: `_posts/` - Posts that are live on the site
- **Draft posts**: `_posts_writing/` - Posts currently being written (not published)
- **Naming convention**: `YYYY-MM-DD-title.md`
- **Front matter** must include: `title`, `date`, `categories`, `tags`
- **Optional front matter**: `math: true` (for LaTeX), `pin: true` (to pin post)

### Pages and Tabs
- **Navigation tabs**: `_tabs/` - Pages that appear in the site navigation
- **Hidden tabs**: `_tabs_hidden/` - Tab pages not shown in navigation but accessible via URL
- **Tab front matter** requires: `layout: page`, `icon`, `order`

### Static Assets
- **Images**: `assets/img/posts/YYYY-MM-DD-title/` - Organized by post date and title
- **Avatar**: `assets/img/avatars/`
- **PDFs**: `pdf/` with subdirectories:
  - `pdf/personal/` - Personal study documents
  - `pdf/college/` - College coursework organized by year
  - `pdf/others/` - Miscellaneous documents

## Configuration

### Site Settings (_config.yml)
- **Language**: Korean (`lang: ko`)
- **Timezone**: Asia/Seoul
- **Theme mode**: Light mode by default
- **Comments**: Uses Giscus (repo: wnsx0000/wnsx0000.github.io)
- **Pagination**: 10 posts per page
- **Collections**: `_tabs` with output enabled

### Important Configuration Notes
- Avoid modifying options below line 155 in `_config.yml` unless necessary
- Permalink structure is `/posts/:title/` - do not change without updating all post links
- Posts default to `layout: post` with comments and TOC enabled

## Deployment

GitHub Actions automatically builds and deploys to GitHub Pages when:
- Pushing to `main` or `master` branch
- Excluding changes to `.gitignore`, `README.md`, `LICENSE`
- Manual workflow dispatch

Workflow: `.github/workflows/pages-deploy.yml`

## Writing Guidelines

### Post Content
- Focus on technical accuracy for AI/ML research papers
- Include Korean explanations with English technical terms in parentheses
- Use LaTeX for mathematical formulas (requires `math: true` in front matter)
- Reference figures with relative paths: `/assets/img/posts/YYYY-MM-DD-title/image.png`

### Front Matter Example
```yaml
---
title: "[Paper Review] Model Name"
date: YYYY-MM-DD HH:MM:SS +0900
categories: [Category, Subcategory]
tags: [ai, ml, specific-topic]
math: true
---
```

### Image Sizing
Use Jekyll's image syntax with width specification:
```markdown
![](/assets/img/posts/post-name/image.png){: width="800"}
```

## Project-Specific Notes

- **No package.json**: This is a Ruby/Jekyll project, not a Node.js project
- **Theme source**: Uses the `jekyll-theme-chirpy` gem, theme files are not in this repo
- **PDF hosting**: The `pdf/` directory hosts academic documents and study materials accessible via the `/documents` page
- **Hidden content**: Use `_tabs_hidden/` for pages that should be accessible but not in main navigation
