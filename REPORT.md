# Delta Twitter support agent: report

## 1. Problem framing

**Brand.** Delta Air Lines. 23,608 threads in the dataset open with a customer tweet and get a Delta reply. 70 percent of those threads are exactly two turns: the customer's tweet and Delta's first reply. That first reply is the unit this agent produces and the unit everything below measures.

**What "good" means for Delta.** Reading 300 random threads made Delta's real operating pattern clear. Delta's public replies almost never resolve anything. They do three things: acknowledge, apologise once, and move the customer to a DM with a confirmation number. Facts (policy, fees, flight status) are stated in public only occasionally and by a human who can check. So for Delta, a good first reply is one that:

1. names the customer's actual problem, not a generic one,
2. asks for the right next thing (confirmation number, arrival airport, flight number, the lost-item form),
3. states no fact it cannot verify and promises nothing, and
4. sounds like Delta: warm, short, one apology.

And a good escalation decision is one that lets the bot post the low-risk acknowledgement on its own, while sending to a human anything where the correct reply needs a fact, a money decision, or sensitive handling. "Escalate" here means "a human reads the tweet before the public reply goes out". It does not mean "a human is eventually involved"; under that definition nearly everything escalates and the metric is useless.

**What was not built, on purpose.**

- No multi-turn handling. The DM conversation that follows the first reply is where Delta actually resolves things, and that data is not in the dataset.
- No factual answering. The agent is not allowed to be the source of policy or schedule facts. Policy questions are always escalated. This is the single biggest design choice and it is discussed in the failure analysis, because the agent still drafts an answer that a human then sees.
- No fine-tuned classifier. A prompted `gpt-4.1-mini` with the taxonomy definitions beat a TF-IDF model by 31 points of macro F1, and a fine-tune would need more labelled data than exists here.
- No Banking77. It has no relation to airline intents.

## 2. System

| stage | method |
|---|---|
| intent | one `gpt-4.1-mini` call with the nine intent definitions, boundary notes, and one example each. Temperature 0. Returns label, confidence, one-line rationale |
| reply | embed the customer tweet with `text-embedding-3-small`, retrieve the 5 most similar historical Delta threads (golden threads excluded from the index), show the model those customers' messages and Delta's real replies, draft in that style under a voice rulebook |
| escalation | deterministic rules first (compensation words, PII patterns, legal or medical words, hostile words, policy intent, unclassifiable intent), then a `gpt-4.1-mini` call that sees the tweet, the intent, the fired rules and the draft. The rules are a floor: the model can add an escalation but cannot remove one |

The whole agent is about 200 lines. The evaluation is about 400.

## 3. Evaluation setup

**Golden set.** 220 first-contact tweets, stratified across intents with a cheap weak label and then labelled by hand from scratch for intent, escalate (yes or no), and escalation reason. How it was drawn and every judgement call is in `data/golden/README.md`.

**Label reliability.** A second annotator, blind to the first labels and working only from the written taxonomy and rules, relabelled 80 of the 220. Agreement: intent kappa 0.94 (95 percent raw), escalation kappa 0.79 (92.5 percent raw). The four intent disagreements were all on the `support_access_issue` versus `policy_or_info_question` versus `loyalty_or_account` boundary. The second annotator was an AI assistant instance with no access to the golden file, not a second person; see section 6.

**Baselines.**

- *trivial*: predict the majority intent, always escalate, reply with one canned Delta line ("We're sorry to hear this. Please follow/DM your confirmation number so we can look into it.").
- *simple*: TF-IDF + logistic regression for intent, trained on 1,780 weak labels (never on golden); reply = Delta's real reply to the nearest historical tweet by TF-IDF cosine; escalation = keyword rules only.

**Metrics.** Intent accuracy and macro F1. Escalation recall (share of must-escalate messages the system escalated), escalation precision, auto rate, and the count of *unsafe autos* (must-escalate messages the system would have posted on its own). Reply quality from an LLM judge (`gpt-5.6-sol`, a different model family from the generator) scoring grounded, correct next step, safe, tone, and an overall 1 to 3 (1 = would embarrass Delta, 2 = fine with light edits, 3 = post as is). The judge sees Delta's real reply as a reference but is told it may be imperfect.

**Judge vs human.** 60 replies (the same 20 golden messages, one reply from each of the three systems, shuffled and shown without the judge's score or the system name) were scored by hand on the overall scale.

| n | Spearman rho | quadratic kappa | exact agreement | within one point |
|---|---|---|---|---|
| 60 | 0.72 | 0.70 | 63 percent | 100 percent |

The judge never disagrees with the human by more than one point. Where it disagrees it is stricter: of the 16 replies the human scored 3, the judge scored 9 of them 2. The judge is therefore a conservative measure of the agent, not a flattering one.

## 4. Results

Results on the 220-example golden set. `make eval` reproduces this table.

| system | intent acc | macro F1 | esc. recall | esc. precision | auto rate | unsafe autos (of 51) | judge overall (1 to 3) | share post-as-is |
|---|---|---|---|---|---|---|---|---|
| trivial | 0.177 | 0.033 | 1.000 | 0.232 | 0.000 | 0 | 1.98 | 0.236 |
| simple | 0.582 | 0.528 | 0.333 | 0.531 | 0.855 | 34 | 1.52 | 0.077 |
| **agent** | **0.841** | **0.839** | **0.941** | 0.649 | 0.664 | **3** | **2.39** | **0.455** |

Reading the table:

- The agent classifies intent correctly 84 percent of the time with balanced per-class performance (every class F1 at or above 0.75; praise and baggage at 0.97). The simple baseline never predicts `other` at all.
- The agent would auto-post two thirds of incoming tweets and would have let 3 of 51 must-escalate messages through. All three are implicit compensation disputes (section 5). Every policy question, every PII leak, every medical, legal or discrimination case and every hostile message was caught.
- The trivial baseline's escalation recall of 1.0 is what always-escalate buys; its auto rate of zero is why it is useless.
- The simple baseline is *worse than trivial* on reply quality (1.52 versus 1.98). Nearest-neighbour replies carry the wrong customer's name, refer to a different problem, and once congratulated a customer who had reported a screaming supervisor. A canned apology is safer than a wrong specific reply.
- The judge scores the agent's replies at 2.39 with 45 percent rated "post as is" and 7 percent rated "would embarrass Delta". On the blind human scoring of 20 agent replies, the human mean was 2.65.

**Escalation recall by reason** (agent):

| reason | n | caught |
|---|---|---|
| needs_factual_answer | 22 | 22 |
| unclear_or_non_english | 10 | 10 |
| compensation_or_refund | 10 | 7 |
| safety_medical_legal_discrimination | 4 | 4 |
| pii_posted_publicly | 3 | 3 |
| hostile_or_threatening | 2 | 2 |

**One iteration, reported honestly.** The first full run had escalation recall 0.804 with 10 unsafe autos. Seven of the ten were tweets the agent correctly classified as `other` (a joke, a phishing warning, a bare link) and then answered anyway, in one case thanking a customer for a phishing report with "we take these scams seriously and will report this right away". One rule was added: `other` always escalates. Recall went to 0.941, unsafe autos to 3, and auto rate fell from 0.723 to 0.664 because five benign `other` tweets ("please hire me and take me away") now go to a human. That trade is the right one for an airline. The before and after summaries are both in `results/`.

## 5. Failure analysis: top five failure modes

**1. The draft answers the factual question it is not allowed to answer.** Policy questions are escalated 22 of 22 times, so no such reply is auto-posted. But the reply is still drafted, and the human reviewer sees it. Of the 15 agent replies the judge scored 1, seven were policy answers with invented specifics:

> *what is the name of the word game on the in-flight panel?* → "The word game on our in-flight panel is called Penny."
> *how long does it take for SPG crossover points to post?* → "typically post within 7 business days"
> *Am I able to travel with a drone and a lithium ion battery fully drained?* → "Fully drained batteries are generally allowed"

The judge's mean on policy questions is 1.82 against 2.82 on praise. Hypothesis: the retrieval examples for policy questions contain Delta's real factual answers to *other* questions, and the model imitates the form (a confident fact) without having the content. Fix: when the intent is `policy_or_info_question`, switch the draft prompt to a "we'll check and DM you" template and never show factual historical replies.

**2. Implicit compensation disputes slip through the rules and the model.** All three remaining unsafe autos:

> "To change to a flight at a different time on the same day ... is going to cost me $525.30?!"
> "RE case 00212762 w bag claims. 4wks passed since I provided garment $val; no response."
> "I am only 33 points from Silver tier and the only remedy is for me to fly again."

None says refund, compensation, or waive. Each is a request for money or status that a human must decide. The compensation regex is keyword-based and the escalation model was told to look for a "disputed charge", which it read narrowly. Hypothesis: the model treats "asks for a DM" as automatically safe, and the rule set lacks the pattern "dollar amount plus complaint". Fix: add a currency-amount trigger and rewrite the compensation rule in the prompt as "the customer wants an outcome that costs Delta money".

**3. Blunt rules escalate praise.** 8 of the 26 false escalations came from the deterministic floor: "Delta hooked it the fuck up" (profanity in praise), "Patients from Shriners Hospitals boarded a 737 ... Santa's Winter Wonderland" (a hospital word in a feel-good story), "7 month pregnant wife" in a seating complaint. These cost a human a look, which is cheap, but they are 4 percent of all traffic. Hypothesis: single-keyword rules cannot see sentiment. Fix: let the praise intent suppress the profanity and medical rules, since praise is the agent's most accurate class (F1 0.97).

**4. The model over-escalates "how do I" as "needs a fact".** 14 false escalations were the escalation model applying `needs_factual_answer` to booking and account messages ("what do I need to do to change my last name in my account", "any aisle seats available on today's Flight 1220"). The gold label is auto because Delta answers these with a DM request. The intent classifier was right; the decision model reinterpreted the message. Hypothesis: the decision prompt describes the trigger by the shape of the question rather than by whether a DM request would answer it. Fix: pass the intent as binding and only allow the model to add the *non-intent* triggers.

**5. Intent boundaries between complaint, disruption and access are soft, and the model and the annotators disagree in the same places.** 14 of the 35 intent errors are `service_complaint_or_feedback` predicted as `flight_disruption` or `support_access_issue`, and the reverse. Example: "had a terrible experience today for my flight from LA-NYC #flt407. 10+ hours and supervisor never showed up" was labelled complaint (past tense, no ask) and predicted disruption. The second human annotator's four disagreements were on the same boundaries. Hypothesis: the taxonomy encodes tense ("in or facing a delay") and the model does not weigh tense. Because both intents are auto with the same reply pattern, this error costs almost nothing downstream, which is an argument for merging them if the classifier is only used for routing. It is not merged here because the split matters for reporting to operations.

A sixth observation that did not make the list because it is a non-finding: the classifier's self-reported confidence is useless. Mean confidence is 0.93 on correct predictions and 0.92 on wrong ones. The `low_confidence` trigger effectively never fires.

## 6. What is misleading about my headline number

The headline is "84 percent intent accuracy, 94 percent escalation recall, 45 percent of replies postable as is". Here is why to distrust each part.

- **The golden set was labelled by the person who wrote the taxonomy and the prompts.** The intent definitions in the classifier prompt and the labelling rules are the same text. Accuracy therefore measures how well the model follows my definitions, not how well my definitions match what Delta's operations team would want. The 0.94 inter-annotator kappa confirms the definitions are clear, not that they are right.
- **The second annotator was not a person.** It was a fresh AI assistant instance given only the written rules and the 80 tweets. This measures whether the rules are followable from the page, which is useful, but it is not a study of human agreement. A real second labeller would very likely agree less, and the escalation kappa of 0.79 is the number I would expect to fall.
- **Escalation recall is measured against my definition of escalate.** Under a stricter definition (for example, "any live disruption needs a human"), recall would be much lower and the auto rate would collapse. The definition is defensible and documented, but it is a choice.
- **The judge is an LLM, and the reference reply it sees is Delta's real reply, which is often bad.** Delta's actual reply to "what is the name of the word game" was about a weather delay to the wrong customer. Judging against that reference is generous in some cases and harsh in others. Agreement with the human was 0.72 Spearman on 60 items, and the human was me, scoring replies produced by prompts I wrote, on a day I had already read the judge's critiques of 15 of them.
- **"Post as is" does not mean correct.** The judge cannot verify facts either. A fluent reply that says "Silver Medallion usually includes a free first checked bag" got a 2 from the judge and a 2 from me; neither of us checked whether it is true for this fare.
- **The stratification used the agent's model family.** Which tweets appear in the golden set was decided by `gpt-4.1-nano`'s weak labels. Tweets that model finds unclassifiable are under-represented in the 58 randomly added rows and over-represented in the `other` bucket, which is exactly where the agent's escalation was weakest.
- **220 examples.** The 95 percent confidence interval on 84 percent accuracy is roughly plus or minus 5 points. Per-reason escalation counts are 2 to 22; "4 of 4 medical cases caught" is not evidence of much.
- **First replies only, 2017 data.** Delta's tone, fees and app have changed. The agent imitates a 2017 support desk.

## 7. What I would do with one more week

1. **Fix failure mode 1 properly**: a separate drafting path for policy questions that never states a fact, then measure whether judge scores on that intent move from 1.8 towards the 2.8 of praise.
2. **Get two real annotators** to label 100 golden items each and report human-human agreement alongside judge-human agreement. Also re-score the 60 replies with a second person.
3. **Build the DM-handoff simulation**: for auto-posted replies, measure whether the DM request was the right next step by checking what Delta actually asked for in the real thread's later turns (the data has 7,000 threads with three or more turns).
4. **Calibrate confidence** by replacing self-reported confidence with agreement between two prompt variants, and test whether disagreement predicts error.
5. **Cost and latency**: the agent makes three model calls and one embedding call per tweet. Merge classify and decide into one call and measure what it costs in escalation recall.
6. **Drift check**: run the agent on the Delta tweets from the last week of the dataset only and compare to the first week, to see whether the retrieval grounding degrades as the vocabulary changes (hurricane season versus Thanksgiving).
