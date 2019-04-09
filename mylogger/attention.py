import html
import os

import numpy
import numpy as np


def viz_sequence(words, scores=None, color="255, 0, 0"):
    text = []

    if scores is None:
        scores = [0] * len(words)
    else:
        # mean = numpy.mean(scores)
        # std = numpy.std(scores)
        # scores = [(x - mean) / (6 * std) for x in scores]

        length = len([x for x in scores if x != 0])
        scores = [x / sum(scores) for x in scores]
        mean = numpy.mean(scores[:length])
        std = numpy.std(scores[:length])

        scores = [max(0, (x - mean) / (6 * std)) for x in scores]

    # score = (score - this.att_mean) / (4 * this.att_std);
    for word, score in zip(words, scores):
        text.append(f"<span class='word' style='background-color: "
                    f"rgba({color}, {score})'>{html.escape(word)}</span>")
    return "".join(text)


def viz_summary(seqs):
    txt = ""
    for name, data, color in seqs:
        if isinstance(data, tuple):
            _text = viz_sequence(data[0], data[1], color=color)
            length = len(data[0])
        else:
            _text = viz_sequence(data)
            length = len(data)

        txt += f"<div class='sentence'>{name}({length}): {_text}</div>"

    return f"<div class='sample'>{txt}</div>"


def sample(words):
    return np.random.dirichlet(np.ones(len(words)))


def samples2dom(samples):
    dom = """
        <!DOCTYPE html>
        <html>
        <head>
        </head>
        <style>
            body{
                    font-size: 14px; /* adjust px to suit */
            }
            .word {
                padding:3px;
                margin: 0px 3px;
                display:inline-block;
            }
            .sentence {
                padding:3px 0;
                float: left;
                width: 100%;
                display:inline-block;
            }
            .sample {
                border:1px solid grey;
                display:block;
                padding:0px 2px;
                display:inline-block;
            }
        </style>
        <body>
        """
    for s in samples:
        dom += viz_summary(s)

    dom += """
        </body>
        </html>
        """
    return dom


def samples2html(samples):
    dom = """
        <style>
            body{
                # font-size: 12px;
                font-family: "CMU Serif", serif;
                font-size: 14px;
            }
            .samples-container {
                background:white;
                background-color:white;
                font-size: 12px;
                color: black; 
            }
            .word {
                padding:2px;
                margin: 0px;
                display:inline-block;
            }
            .sentence {
                padding:2px 0;
                float: left;
                width: 100%;
                display:inline-block;
            }
            .sample {
                border:1px solid grey;
                display:block;
                padding:0px 4px;
                display:inline-block;
            }
        </style>
        <div class='samples-container'>
        """

    for s in samples:
        dom += viz_summary(s)

    dom += """
        </div>
        """
    return dom


def viz_seq3(dom):
    # or simply save in an html file and open in browser
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'attention.html')

    with open(file, 'w') as f:
        f.write(dom)

# samples = []
# for i in range(10):
#     source = lorem.sentence().split()
#     scores = sample(source)
#     summary = lorem.sentence().split()
#     reconstruction = lorem.sentence().split()
#     samples.append(((source, scores), summary, reconstruction))
# viz_seq3(samples2html(samples))
