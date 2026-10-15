"""Single entry. Read this file from top to bottom.

1. loading.py loads the local Qwen3.5-0.8B snapshot.
2. configuration.py holds the width and the six decision-token ids.
3. decision/rows.py rewrites only those six embedding rows.
4. processing.py builds one shared prefix and one suffix per option.
5. modeling.py runs the prefix once, forks each option, then scores it.
   The fork and the score live in decision/branch.py and decision/score.py.

Run from the simpleJev directory:

    python -m simplejev
"""

from .configuration import SimpleJevConfig
from .loading import load_backbone
from .modeling import SimpleJevForDecision
from .processing import SimpleJevProcessor

QUESTION = "一加一等于几？"
OPTIONS = ["二", "三", "四"]


def main() -> None:
    backbone, processor = load_backbone()
    config = SimpleJevConfig()
    model = SimpleJevForDecision(config, backbone)

    model.init_decision_rows()
    encoded = SimpleJevProcessor(processor.tokenizer, config)(QUESTION, OPTIONS)
    probs = model(encoded.prefix_ids, encoded.suffix_ids).probs
    print(probs, float(probs.sum()))


if __name__ == "__main__":
    main()
