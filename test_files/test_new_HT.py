from river import synth
from river import evaluate
from river import metrics
from river import tree

gen = synth.Agrawal(classification_function=0, seed=42)
# Take 1000 instances from the infinite data generator
dataset = iter(gen.take(1000))
print(dataset)

model = tree.HoeffdingTreeClassifier(grace_period=100,split_confidence=1e-5,nominal_attributes=['elevel', 'car', 'zipcode'])

metric = metrics.Accuracy()

evaluate.progressive_val_score(dataset, model, metric)
