## Beehive Sound Data Exploration and Health Monitoring
 
## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Members](#project-members)
4. [Environment Setup](#environment-setup)
5. [Methods](#methods)
6. [Results](#results)
7. [Figures](#figures)
8. [Discussion](#discussion)
9. [Conclusion](#conclusion)
10. [Statement of Collaboration](#statement-of-collaboration)
11. [Notebooks](#notebooks)
12. [Extra Credit](#extra-credit)
---
 
## Introduction
 
**We tried to work out whether a beehive is healthy just by listening to it.**
 
Bees pollinate about a third of the world's food. Keeping them healthy matters, but
checking on a hive is a hassle — a beekeeper has to physically open it up, which
takes time and disturbs the bees.
 
Here's the useful part: **a hive sounds different when something is wrong.** The
buzzing changes pitch and intensity when the colony loses its queen, gets sick, or
is about to swarm. So if a computer could listen to a hive and tell you what's
happening inside, a beekeeper could keep an eye on hundreds of hives at once
without opening a single one.
 
That's what we built: a model that listens to a recording and guesses which of six
health conditions the hive is in.
 
## Why we needed big data tools
 
Two reasons.
 
**The recordings are huge.** 23.21 GB of audio files. That's more than a normal
computer can hold in memory at once.
 
**Processing audio is slow.** Before a computer can learn from a sound file, the
sound has to be turned into numbers. Doing that to thousands of files one at a
time would take hours.
 
So we used **Spark** on **SDSC Expanse**, a supercomputer, which let us split the
work across **15 workers running at the same time**. Without that, we would have
had to throw away most of the recordings and work with a small sample.
 
---
 
## Dataset
 
**Dataset:** [Smart Bee Colony Monitor: Clips of Beehive Sounds](https://www.kaggle.com/datasets/annajyang/beehive-sounds)
 
The dataset contains recordings of beehives, plus sensor readings from inside each
hive (temperature, humidity, pressure) and the weather outside at the time.
 
Every recording comes with a label: which of **six health conditions** the hive was
in, numbered 0 to 5.
 
The raw audio is **23.21 GB**. Once we turned each recording into a row of numbers,
we ended up with about **1,275 labelled examples** to train on.
 
---
 
## Project Members
 
- **Adham Kamel** — adkamel@ucsd.edu
- **Snigdha Tiwari** — sntiwari@ucsd.edu
- **Patcharapol Puckdee** — ppuckdee@ucsd.edu
- **Conner Houghtby** — choughtby@ucsd.edu
---
 
## Environment Setup
 
We logged into SDSC Expanse through [portal.expanse.sdsc.edu](https://portal.expanse.sdsc.edu)
and started a JupyterLab session with these settings:
 
| Field | Value |
|---|---|
| Account | `TG-SEE260003` |
| Partition | `shared` |
| Number of Cores | 16 |
| Memory (GB) | 32 |
| Singularity Image | `~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif` |
| Environment Modules | `singularitypro` |
| Type | JupyterLab |
 
## Spark settings
 
```python
spark = SparkSession.builder \
    .master("local[15]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.instances", 15) \
    .getOrCreate()
```
 
## Why we asked for this much
 
We requested **16 cores** and **32 GB of memory** for the 23 GB of audio.
 
Think of it as hiring a team. One worker is the **manager** (the "driver") who hands
out jobs. The other 15 are **workers** (the "executors") who actually do them. The
memory gets divided between them:
 
- **Manager:** 2 GB
- **Number of workers:** 16 − 1 = **15**
- **Memory each worker gets:** (32 GB − 2 GB) ÷ 15 = **2 GB each**
We picked these numbers because:
 
- The audio is about 23 GB, which is far too much for one machine
- Turning sound into numbers is slow, repetitive work — exactly the kind of thing
  that gets faster when you split it across many workers
- 2 GB per worker gives each one enough room to handle an audio file
---
 
## Methods
 
## Looking at the data first
 
Before building anything, we loaded the spreadsheet of hive information and checked
what was actually in it. Three things stood out:
 
**One column was mostly empty.** `gust_speed` (wind gusts) was missing for **994
rows, about 78% of the data**. There wasn't enough left to be useful, so we deleted
the column.
 
**Four columns had a few gaps.** `weather_temp`, `wind_speed`, `lat` and `long` were
each missing **4 values**. That's tiny, so we filled the gaps with the average of
each column.
 
**The health conditions were very uneven.** Condition 3 showed up far more often
than the others, while conditions 0 and 2 were rare. This matters more than it
sounds — see below.
 
## Getting the data ready
 
**Turning sound into numbers.** A computer can't learn from an audio file directly.
We used a tool called `librosa` to measure things about each recording: how loud it
is, how high or low the pitch is, and how the sound energy is spread across
different frequencies. Each recording came out as a row of numbers describing what
it sounds like.
 
**Adding everything else.** We attached the hive sensor readings and the weather
data to each recording, so the model could use those too.
 
**Putting it all on the same scale.** Some measurements are big numbers and some are
tiny. Left alone, the big ones would drown out the small ones just because of their
size, so we rescaled everything to be comparable.
 
```python
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler    = StandardScaler(inputCol="features", outputCol="features_scaled",
                           withMean=True, withStd=True)
pipeline  = Pipeline(stages=[assembler, scaler])
```
 
**Fixing the uneven conditions.** This one is important. If condition 3 makes up
most of the data, a lazy model learns to just guess "condition 3" every time. It
scores well and is completely useless. To stop that, we told the model that
mistakes on the rare conditions count for more, which forces it to pay attention to
them.
 
**Splitting the data.** We saved the processed data and split it **80/20**:
**1,053 examples to learn from** and **222 held back for testing**. The model never
sees the test set while learning, so testing on it shows whether it actually learned
something or just memorised the answers.
 
## Model 1: Decision Tree
 
A decision tree is basically a **flowchart of yes/no questions**. "Is the
temperature above 30 degrees? Is the buzzing higher-pitched than this?" Follow the
answers down and you arrive at a guess.
 
The main setting is how many questions deep the flowchart is allowed to go. We
tried depths of **1, 2, 3, 5, 7, 10 and 15** to find the sweet spot.
 
```python
dt = DecisionTreeClassifier(
    labelCol="target",
    featuresCol="features",
    maxDepth=5,
    weightCol="weight"
)
```
 
## Model 2: PCA + Logistic Regression
 
This one has two steps.
 
**PCA** takes a large pile of measurements and **squashes them into a smaller pile**,
keeping the information that actually varies between recordings and dropping the
noise. Lots of our sound measurements overlap and say similar things, so this
tidies them up.
 
**Logistic regression** then takes that tidier input and draws boundaries between
the six health conditions.
 
We first ran PCA at 20 components to see how the information was spread out, then
settled on **10** as our main setting. We also tried 2, 5, 10, 15 and 20 to see how
the choice affected accuracy.
 
```python
pca = PCA(k=10, inputCol="features", outputCol="pca_features")
lr  = LogisticRegression(
    featuresCol="pca_features",
    labelCol="target",
    weightCol="weight",
    maxIter=100
)
pipeline = Pipeline(stages=[pca, lr])
```
 
We tried this after the decision tree because clearing out the noisy, overlapping
measurements first makes it harder for the model to memorise the training data
instead of learning real patterns.
 
## Model 3: Looking for natural sound patterns
 
The two models above are told the right answer while they learn. This third
analysis asks a different question: **if you don't tell the computer anything, do
the recordings naturally fall into groups?**
 
The idea is that a hive might have a kind of "sound language" — recurring
combinations of loudness, pitch and texture that mean something, even before anyone
labels them.
 
To keep this honest we used **only the sound measurements**, nothing about which
hive it was or when it was recorded, so the groups couldn't form around those
instead of the audio. We squashed the sound measurements down with PCA, then used
**KMeans**, which sorts things into a set number of groups based on how similar
they are.
 
Afterwards we compared the groups against the real health labels:
 
- If a group is mostly **one health condition**, that condition has a recognisable
  sound
- If groups contain a **mix of conditions**, the sounds overlap too much, and we'd
  need better or longer recordings to tell them apart
---
 
## Results
 
## Model 1: Decision Tree
 
At depth 5 the flowchart did well. Its score on the data it had seen and the data
it hadn't were within **3–4%** of each other, which means it was learning real
patterns rather than memorising.
 
When we let the flowchart get deeper, it got near-perfect on the training data
while its test score stopped improving. **That's the classic sign of memorising**
rather than learning. We picked the depth with the best test score.
 
## Model 2: PCA + Logistic Regression
 
## How much information each component keeps
 
PCA squashes many measurements into fewer. This shows how much of the original
information survives.
 
| Components (k) | Information kept by this one | Running total |
|---|---|---|
| 1 | 17.35% | 17.35% |
| 2 | 12.07% | 29.42% |
| 5 | 5.95% | 52.63% |
| 10 | 2.90% | 73.85% |
| 15 | — | 86.16% |
| 20 | — | ~93% |
 
## How well it scored (using 10 components)
 
<img width="470" height="74" alt="model2metrics" src="https://github.com/user-attachments/assets/f3f7a8ef-6551-4fa7-8c53-d6da8bb1f0fb" />
|  | Data it learned from | Data it had never seen | Difference |
|---|---|---|---|
| Accuracy | 0.7056 | 0.7117 | −0.006 |
| F1 (weighted) | 0.7122 | 0.7181 | −0.006 |
| Error Rate | 0.2944 | 0.2883 | −0.006 |
 
**About 71% correct**, and — this is the good part — it scored the *same* on new
data as on data it had already seen. That means it genuinely learned rather than
memorised.
 
## Does using more components help?
 
| Components (k) | Errors on training data | Errors on new data |
|---|---|---|
| 2 | 0.3599 | 0.4099 |
| 5 | 0.3485 | 0.3964 |
| 10 | 0.2944 | 0.2883 |
| 15 | 0.1985 | 0.2252 |
| 20 | 0.1311 | 0.1667 |
 
More components kept helping. The **best result was 20 components (16.67% wrong)**.
We used 10 as our main model, since it keeps most of the useful information with a
simpler setup.
 
<img width="790" height="490" alt="PCA Logistic Regression training vs test error graph" src="https://github.com/user-attachments/assets/c52720b8-d80a-4e50-8a39-f68ce3cb73f4" />
## What it got right and wrong
 
This grid shows every guess on the 222 test recordings. **Each row is what the hive
actually was; each column is what the model guessed.** Numbers along the diagonal
are correct — everywhere else is a mistake.
 
| Actual \ Guessed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| **0** | 11 | 1 | 1 | 1 | 3 | 4 |
| **1** | 0 | 9 | 4 | 2 | 0 | 0 |
| **2** | 0 | 4 | 10 | 0 | 0 | 0 |
| **3** | 5 | 6 | 3 | 57 | 0 | 0 |
| **4** | 1 | 0 | 0 | 1 | 20 | 12 |
| **5** | 6 | 0 | 0 | 1 | 9 | 51 |
 
**What went well:** conditions 3 (57 out of 71), 5 (51 out of 67) and 4 (20 out of
34) were the most reliable. These are the common conditions, so the model had the
most examples to learn from.
 
**The serious problem:** condition 0 was **missed 10 times out of 21** — nearly
half. Condition 0 means an unhealthy hive, which is the whole point of the system.
A tool that misses half the sick hives isn't much of an alarm.
 
**The other problem:** conditions 4 and 5 kept getting mixed up with each other (12
mistakes one way, 9 the other). They probably just sound alike.
 
## Did more workers make it faster?
 
We timed one part of the pipeline using different numbers of workers. Each number
is the average of 3 runs, with the first thrown away to let the system warm up.
 
| Workers | Memory each | Time (s) | How much faster | Efficiency |
|---|---|---|---|---|
| 1 | 64 GB | 0.80 | 1.00× | 100% |
| 3 | 20 GB | 0.77 | 1.03× | 34% |
| 7 | 14 GB | 0.78 | 1.03× | 15% |
 
Barely any difference. Doing the maths, only about **3.4% of that step can actually
be split up** — the rest has to happen in order.
 
It's like four people carrying one couch. Three of them end up standing around,
because the job doesn't divide.
 
<img width="1289" height="495" alt="strong scaling analysis graph" src="https://github.com/user-attachments/assets/3038a522-c872-4317-a846-5e178693c632" />
---
 
## Figures
 
| Figure | Location | What it shows |
|---|---|---|
| PCA Explained Variance | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | How much information is kept as you add components (1–20) |
| Model 2 Metrics Table | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Scores on seen and unseen data at 10 components |
| Fitting Curve (Model 2) | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | How errors change across 2, 5, 10, 15 and 20 components |
| Confusion Matrix (Model 2) | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Every guess the model made on the test recordings |
| Fitting Curve (Model 1) | [model1_decision_tree.ipynb](model1_decision_tree.ipynb) | How errors change as the flowchart gets deeper |
| Strong Scaling Plot | [model2_pca_logistic_regression.ipynb](model2_pca_logistic_regression.ipynb) | Actual speed gain from extra workers vs. what was predicted |
 
<img width="889" height="490" alt="PCA explained variance plot" src="https://github.com/user-attachments/assets/5b63a35a-0f0d-43a4-95ac-a65629215365" />
---
 
## Discussion
 
## The data
 
Deleting `gust_speed` was easy to justify — with 78% of it missing there simply
wasn't enough to work with. The other four columns were only missing 4 values each,
so filling those with the average won't have changed anything meaningful.
 
**The uneven health conditions were the real challenge.** Without our weighting fix,
both models would have leaned heavily toward guessing condition 3 and largely
ignored the rare ones. The weighting helps, but it can't invent examples that don't
exist — some conditions just have very little data behind them.
 
## Model 1: Decision Tree
 
The flowchart worked well when kept short, which makes sense: things like
temperature and humidity have natural cut-off points, and yes/no questions handle
those neatly.
 
**The problem with deeper flowcharts** is what they start paying attention to. Given
enough questions, the model can latch onto *which specific hive* a recording came
from, or *what time of day* it was recorded, instead of the sound itself. That works
on our hives and would fall apart on anyone else's. Removing those identifying
details would give a more honest picture of how good the model really is.
 
## Model 2: PCA + Logistic Regression
 
At 10 components the model is in a good place — it scored the same on new data as on
old (a difference of just −0.006). That's what you want. It makes sense too: with
only about 1,275 examples and 10 simplified inputs, there isn't much room to
memorise.
 
Adding more components kept improving things, with 20 giving the best result. That
suggests there's more to squeeze out, but we'd want more recordings before pushing
much further.
 
**The confusion grid points to two weaknesses.**
 
Conditions 4 and 5 get mixed up constantly, which suggests they sound genuinely
similar and our model draws boundaries that are too simple to separate them.
 
Condition 0 has the worst miss rate — only 11 of 21 caught. **In the real world this
is the failure that matters**, because missing an unhealthy hive defeats the purpose
of building the thing. A more flexible model would likely handle both problems
better.
 
**Compared to Model 1**, this model is steadier. The decision tree's gap between
seen and unseen data grew as it got deeper, while this one stayed near zero
throughout.
 
## Speed
 
Extra workers barely helped — 1.03× faster with either 3 or 7. That's because the
step we timed (loading a file and summarising it) mostly has to happen in order.
 
**But we measured the wrong step.** Loading a spreadsheet is quick and sequential.
The genuinely slow part is processing thousands of audio files, and that *does*
split up well, because each file can be handled independently. We didn't time that
step separately, so these numbers make Spark look less useful than it actually was
for this project.
 
---
 
## Conclusion
 
We set out to work out a beehive's health from its sound, using a supercomputer to
handle the data.
 
**Model 1 — Decision Tree:** Worked well when kept simple, and it's easy to
understand what it's doing. But it starts memorising when allowed to get complex,
and may have been relying on which hive a recording came from rather than the sound
itself.
 
**Model 2 — PCA + Logistic Regression:** Got about **71% correct**, with no gap
between how it did on familiar and unfamiliar data. Its weaknesses were mixing up
conditions 4 and 5, and missing too many condition 0 hives — the ones that matter
most.
 
**What we learned about big data:** More workers doesn't automatically mean faster.
Some jobs split up beautifully and some barely at all, and **knowing which is which
matters more than knowing how to use the tools.** Our own speed test measured a step
that couldn't be split, which is exactly why we got such a disappointing result.
 
**Why the supercomputer mattered:** Without it we'd have been stuck working with a
small sample of the recordings. Processing all 23 GB gave us an accurate picture of
how common each health condition is, and let us test many settings in one session.
 
**What we'd try with more time:**
 
- More flexible models to separate conditions 4 and 5
- Richer ways of describing sound, or a model already trained on audio
- More components, with safeguards against memorising
- Making condition 0 mistakes cost more, since missing a sick hive is the worst
  outcome
- Timing the audio processing step specifically, where extra workers should
  genuinely help
---
 
## Notebooks
 
| Notebook | What's in it |
|---|---|
| [Data Exploration](data_exploration.ipynb) | First look at the data: what's in it, what's missing, how common each condition is |
| [Preprocessing + Decision Tree (Model 1)](model1_decision_tree.ipynb) | Turning sound into numbers, preparing the data, and building Model 1 |
| [PCA + Logistic Regression (Model 2)](model2_pca_logistic_regression.ipynb) | Simplifying the data, building Model 2, testing it, and the speed experiment |
| [Extra Credit: Spark vs Ray](extra_credit.ipynb) | Comparing two big data tools on the same job |
 
## Extra Credit
 
**Which tool was faster?**
 
Spark, by a wide margin — **1.01 seconds against Ray's 142.94 seconds.** The job was
summarising a spreadsheet, which Spark does in a single pass. Ray reads the file
from start to finish before splitting up the work, and at this size that setup cost
outweighs any benefit from splitting.
 
**Which was easier to write?**
 
Spark. It has built-in commands that produce a full summary in one line. Ray has no
built-in way to calculate percentiles, so we had to convert the data to a different
format first, which meant extra steps and messier code.
 
**Which would we pick?**
 
**Spark**, for this kind of spreadsheet work and data preparation.
 
**Ray** would be the better pick for the audio processing at large scale, or for
training deep learning models, since it has stronger tools for both.
 
---
 
## Statement of Collaboration
 
- **Snigdha Tiwari** — Contributor: data exploration, preprocessing, final repo
  cleanup, extra credit
- **Adham Kamel** — Contributor: data visualisation, training the Decision Tree
  model, and the speed analysis for the PCA model
- **Patcharapol Puckdee** — Contributor: documentation, including the README and
  write-up, and supporting team collaboration through feedback and communication
- **Conner Houghtby** — Contributor: <!-- TODO: this entry is incomplete -->
