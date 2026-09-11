# Build LaTeX and PDF from the Markdown sources in this directory.
#
#   make                          build every PDF
#   make posterior_uncertainty.pdf   build one
#   make tex                      regenerate the .tex files only
#   make clean                    remove build artifacts (keeps .tex/.pdf)
#   make distclean                also remove the generated .tex and .pdf
#
# Requires pandoc and a LaTeX distribution providing latexmk + pdflatex.
# Only the .md and .tex are tracked in git; the PDF is a build product.

MD  := $(wildcard *.md)
TEX := $(MD:.md=.tex)
PDF := $(MD:.md=.pdf)

BUILDDIR := .build

PANDOC       := pandoc
PANDOC_FROM  := gfm+tex_math_dollars
PANDOC_FLAGS := --standalone --toc --toc-depth=2 --highlight-style=tango \
                -V documentclass=article -V fontsize=11pt \
                -V geometry:margin=1in -V colorlinks=true \
                -V linkcolor=NavyBlue -V urlcolor=NavyBlue

# pandoc escapes a raw \today into literal text, so format the date here.
BUILDDATE := $(shell date '+%-d %B %Y')

LATEXMK       := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILDDIR)

.PHONY: all tex clean distclean
.SECONDEXPANSION:
.DEFAULT_GOAL := all

all: $(PDF)

tex: $(TEX)

# The first line of each .md is its H1 title. Strip it and promote the text
# to document metadata so pandoc typesets a real title block instead of a
# section heading. A sibling <name>.meta.yaml, if present, supplies the
# title/subtitle/author; -M beats --metadata-file in pandoc, so the H1 is
# used only as the fallback title when no meta file exists.
%.tex: %.md $$(wildcard $$*.meta.yaml)
	@echo "  PANDOC  $@"
	@if [ -f $*.meta.yaml ]; then \
	    title=""; meta="--metadata-file=$*.meta.yaml"; \
	else \
	    title="-M title=$$(sed -n '1s/^#\{1,\} *//p' $< | tr '\n' ' ')"; meta=""; \
	fi; \
	tail -n +2 $< | $(PANDOC) - -f $(PANDOC_FROM) -t latex $(PANDOC_FLAGS) \
	    -M date="$(BUILDDATE)" $$meta $$title -o $@

# Aux files stay in $(BUILDDIR); only the PDF is lifted back out.
%.pdf: %.tex
	@echo "  LATEXMK $@"
	@mkdir -p $(BUILDDIR)
	@$(LATEXMK) $(LATEXMK_FLAGS) $< > $(BUILDDIR)/$*.build.log 2>&1 \
	    || { echo "latexmk failed; see $(BUILDDIR)/$*.build.log"; exit 1; }
	@cp $(BUILDDIR)/$@ $@

clean:
	rm -rf $(BUILDDIR)

distclean: clean
	rm -f $(TEX) $(PDF)
