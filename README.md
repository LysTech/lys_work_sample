# Lys Work Sample

The task consists in classifying brain recordings: we have provided a baseline, and the goal is to improve upon it.

## Experimental Setup

The participant was told to think one of three things: (a) sing a particular song in your head (always the same one), (b) imagine using WhatsApp (always in the same way), (c) perform a mental arithmetic task of your choosing (these were always multiplications, examples: 15x37, 62x19, …). The cortical activity was then recorded with an [fNIRS](https://en.wikipedia.org/wiki/Functional_near-infrared_spectroscopy) scanner, in particular, the [Kernel Flow2](https://www.kernel.com/specs/Flow%202%20Spec%20Sheet.pdf). We provide the data, already processed, and a simple classification script.

Download [the data here](https://drive.google.com/drive/folders/1gKdTGaIYbZ8yBP7q3RsSMRtRyI-lHmPM?usp=sharing), and view the script in this repo. 

Peak RAM usage is ~6.5GB, if this is an issue, feel free to rent a server online (e.g. via Paperspace or AWS) and we’ll reimburse your usage costs. The use of memmap and accompanying complexity is intended for RAM usage minimisation.

## Task

The first sub-task counts for 80% of the marks, the second one for the other 20%.

1. Improve the overall classification accuracy of this script. Full marks corresponds to 80% or more classification accuracy.

2. Improve the baseline script, by making it more memory efficient, faster or simpler. We leave it to you to decide what is most interesting or worthwhile here.

```

Overall: 64.6% +/- 6.8%

Chance level: 33.3%

Per-fold results:

  Fold 1 (sessions [1, 3]): 68.5%

  Fold 2 (sessions [4, 5]): 65.4%

  Fold 3 (sessions [6, 7]): 70.1%

  Fold 4 (sessions [9, 11]): 51.2%

  Fold 5 (sessions [12, 13]): 61.4%

  Fold 6 (sessions [14, 15]): 70.9%

Per-class accuracy (averaged across folds):

  Mental Arithmetic: 82.6% +/- 6.7%

  Singing a song: 63.4% +/- 18.0%

  WhatsApp: 44.8% +/- 7.3%

```

### Some definitions

Trial: a single instance of the participant thinking about X is a trial. 

Stimulus: A bit of a misnomer, means “X” if the participant was thinking about X during the trial. In this example, this could be either “singing a song”, “whatsapp”, or “mental arithmetic”.

GLM: general linear model.

HRF: [Haemodynamic response function.](https://en.wikipedia.org/wiki/Haemodynamic_response)
