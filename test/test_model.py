import model
import candidate

tableaux = candidate.TableauSet('test/bata.txt')
maxent_model = model.MaximumEntropy(tableaux)

def test_batch_learn_variable():
    constraints_and_weights = maxent_model.batch_learn_variable()
    constraints, weights = zip(*constraints_and_weights)
    assert weights[0] > weights[1]

def test_get_distribution():
    weights = [1, 0.5]
    cands_probs = maxent_model.get_distribution(weights)
    cands, probs = zip(*cands_probs)
    assert probs[0] > probs[1]
    assert probs[2] > probs[3]
    assert probs[4] > probs[5]

def test_batch_learn_categorical():
    categories = [0,1,0,1,0,1] # 0 is in
    constraints_and_weights = maxent_model.batch_learn_categorical(categories)
    assert constraints_and_weights[0][1] > constraints_and_weights[1][1]
