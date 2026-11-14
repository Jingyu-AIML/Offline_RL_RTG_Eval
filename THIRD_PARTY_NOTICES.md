# Third-party notices and research credits

This package's contributions are distributed under the [MIT License](LICENSE).
The upstream notice below is retained separately. Redistribution must preserve
the applicable copyright and permission notices.

## Decision Transformer

The Decision Transformer portion of [src/model.py](src/model.py) was documented
as adapted from a MuJoCo Decision Transformer implementation. That description
did not record an exact source repository or commit. The official implementation
is credited here and its complete MIT notice is retained; this does not assert
a verified commit-level lineage.

- Paper: Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover,
  Michael Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch (2021).
  [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).
- Official implementation: [kzl/decision-transformer](https://github.com/kzl/decision-transformer),
  including its [Gym model](https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py).
- License source: [LICENSE.md](https://github.com/kzl/decision-transformer/blob/master/LICENSE.md).

This package adapts the model to discrete genre prediction with cross-entropy,
rating-based returns, local transformer blocks, and padding-aware contexts.
The following notice is reproduced verbatim from the official implementation:

```text
MIT License

Copyright (c) 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors (https://arxiv.org/abs/2106.01345)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## SASRec research credit

The SASRec baseline in [src/model.py](src/model.py) follows the self-attentive
sequential recommendation approach of Wang-Cheng Kang and Julian McAuley (2018),
[Self-Attentive Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/icdm18.pdf),
ICDM 2018. The authors' reference implementation is
[kang205/SASRec](https://github.com/kang205/SASRec).

The baseline here uses the package's PyTorch transformer blocks to predict
genres. This citation credits the research method; the authors' reference
implementation is not bundled in this package.

## External dependencies and datasets

The libraries listed in [requirements.txt](requirements.txt) are installed
separately and remain subject to their own licenses and notices. Their source
code is not vendored in this package.

MovieLens 25M and MyAnimeList Database 2020 are not bundled. Obtain them from the
providers linked in the [README](README.md#prepare-the-data) and follow the
providers' applicable terms. This package's MIT license does not relicense those
datasets or the external libraries.
