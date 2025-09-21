# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Jekyll-based academic website using the **al-folio** theme. It's a static site generator that creates personal academic websites with features for publications, projects, blog posts, CV, and more.

## Development Commands

### Local Development

- **Start development server (Docker - Recommended):**

  ```bash
  docker compose pull && docker compose up
  ```

  Site will be available at `http://localhost:8080`

- **Alternative development (Docker Slim):**

  ```bash
  docker compose -f docker-compose-slim.yml up
  ```

- **Legacy local setup (not recommended):**
  ```bash
  bundle install
  bundle exec jekyll serve --watch --port=8080 --host=0.0.0.0 --livereload
  ```

### Build and Deploy

- **Build for production:**

  ```bash
  bundle exec jekyll build
  ```

- **Deploy (handled automatically via GitHub Actions on push to main)**

### Code Quality

- **Format code with Prettier:**

  ```bash
  npx prettier --write .
  ```

- **Run pre-commit hooks:**
  ```bash
  pre-commit run --all-files
  ```

### Testing

- **Check for broken links (requires built site):**
  ```bash
  bundle exec htmlproofer ./_site --disable-external
  ```

## Architecture

### Key Directories

- `_config.yml` - Main Jekyll configuration file
- `_layouts/` - Page layout templates (Liquid)
- `_includes/` - Reusable template components
- `_pages/` - Static pages (about, CV, publications, etc.)
- `_posts/` - Blog posts
- `_bibliography/` - BibTeX files for publications
- `_data/` - YAML data files
- `_sass/` - SCSS stylesheets
- `assets/` - Images, PDFs, and other static assets

### Content Collections

- **Publications:** Managed through `_bibliography/papers.bib` using Jekyll Scholar plugin
- **Projects:** Stored in `_projects/` directory
- **Books:** Book reviews and reading list in `_books/`
- **News:** Short updates in `_news/`
- **Blog posts:** Long-form content in `_posts/`

### Key Technologies

- **Jekyll** - Static site generator
- **Liquid** - Templating language
- **Bootstrap** - CSS framework
- **MathJax** - Math rendering
- **Jekyll Scholar** - Bibliography management
- **Multiple Jekyll plugins** - See Gemfile for full list

### Custom Features

- Bibliography with citation metrics (Altmetric, Dimensions, Google Scholar)
- Responsive image handling with ImageMagick
- Dark/light mode toggle
- Search functionality
- Social media integration
- Docker containerization

## Configuration

### Main Settings

Most site configuration is in `_config.yml`:

- Personal information (name, contact, description)
- Theme settings and colors
- Plugin configurations
- Collection settings
- Social media links
- Analytics setup

### Content Management

- **Publications:** Add entries to `_bibliography/papers.bib`
- **Projects:** Create markdown files in `_projects/`
- **Blog posts:** Create markdown files in `_posts/` with YYYY-MM-DD-title.md format
- **Pages:** Edit existing files in `_pages/` or create new ones

## Development Notes

### File Naming Conventions

- Blog posts: `YYYY-MM-DD-title.md`
- Projects: `project-name.md`
- Pages: descriptive names like `about.md`, `publications.md`

### Image Handling

- Place images in `assets/img/`
- ImageMagick automatically generates responsive versions
- Supports WebP conversion for performance

### Bibliography Management

- Uses Jekyll Scholar for publication rendering
- Supports various BibTeX fields: `pdf`, `code`, `website`, `arxiv`, etc.
- Automatic citation metrics integration

### Styling

- SCSS files in `_sass/`
- Bootstrap-based with custom theme colors
- Responsive design with mobile support

### Docker Development

- Preferred development method
- Handles Ruby/Jekyll dependencies automatically
- Live reload enabled
- Configuration file changes trigger container restart
