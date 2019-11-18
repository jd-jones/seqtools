# seqtools
Utilities for structured prediction of sequences, built on [pytorch-struct] and [mfst].

**NOTE:** This is an early-stage project under active development, and is subject to change without notice.

## Installation
Clone this repo and then install it with pip. Installation will go about 5 minutes without printing anything at some point, while it is installing `mfst`.

**NOTE:** I've tried installing ``mfst`` on OSX, but it crashes because of some problem with LLVM (I think).

**ONE MORE THING:** I haven't tried installing this fom scratch yet, so I'm not sure if all the dependencies install properly.

``` console
user@host:~$ pip install -e /PATH/TO/seqtools/
```

## Usage
See `tests` for usage examples (but there are none right now).

[pytorch-struct]: https://github.com/harvardnlp/pytorch-struct
[mfst]: https://github.com/matthewfl/openfst-wrapper
