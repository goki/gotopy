# GoToPy

GoToPy is a Go to Python converter -- translates Go code into Python code.

It is based on the Go `gofmt` command source code and the go `printer` package, which parses Go files and writes them out according to standard go formatting.

We have modified the `printer` code in the `pyprint` package to instead print out Python code.

# TODO

* add gopy flag

* strings.Fields(x) -> x.split()

* class comments -> """

* switch -> ifs.. -- grab switch expr and put into each if

# gopy specific mode

* replace []string() -> go.Slice_string etc 

# Check


