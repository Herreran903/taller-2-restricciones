pdf:
	latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex

clean:
	latexmk -C

veryclean:
	latexmk -C
	rm -f main.pdf

