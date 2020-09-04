# GoToPy

GoToPy is a Go to Python converter -- translates Go code into Python code.

It is based on the Go `gofmt` command source code and the go `printer` package, which parses Go files and writes them out according to standard go formatting.

We have modified the `printer` code in the `pyprint` package to instead print out Python code.

