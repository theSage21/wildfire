TorchZoo
========

Implementations of various NNs in pytorch


```python
# simple models
from torchzoo import FeedForward, COOL

# recurrent models
from torchzoo import RWA
```

Benchmarks / Reproduction studies are held in [TorchCircus](https://github.com/theSage21/torchcircus).


How to use Zoo/Circus
-------

1. Finding a paper
    - all book keeping is done in the github Issues.
    - Since most papers end up on arxiv, search for an arxiv ID like `1805.10369`
    - If the paper is not on Arxiv, search for the full title of the paper.
3. The [issue labels](https://github.com/theSage21/torchzoo/labels) provide a mechanism to find specific things.
4. If a paper passes it's reproduction in the Circus, the underlying CORE parts are put into the [master](https://github.com/theSage21/torchzoo/tree/master) branch.
5. If a paper has failed in it's reproduction, the core parts are sent to the [fertilizer](https://github.com/theSage21/torchzoo/tree/graveyard) branch.
6. If something is still not covered, please open an issue.


I'll try to look into things whenever I have time.
