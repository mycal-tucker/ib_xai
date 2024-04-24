import numpy as np
from embo import InformationBottleneck
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle as pkl


def _2d_plot(embs, values):
    # Plot PCA of the embeddings:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embs)
    # Create a scatter plot of the PCA result
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=values[:, 0], cmap='viridis')
    plt.title('2D PCA Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Scalar Values')
    plt.savefig('data/embs/pca')
    plt.close('all')


def _load_single_suite(suite_dir):
    text = []
    with open('data/embs/%s/text.txt' % suite_dir, 'r') as f:
        for line in f:
            text.append(line)
    with open('data/embs/%s/embs.pkl' % suite_dir, 'rb') as f:
        embeddings = pkl.load(f)
    with open('data/embs/%s/values.pkl' % suite_dir, 'rb') as f:
        values = pkl.load(f)
    print(text)
    return text, embeddings, values


def load_data():
    suites = ['suite_demo', 'suite_poison_demo']
    text = []
    embeddings = []
    values = []
    for suite in suites:
        print("Loading suite", suite)
        _text, _embed, _values = _load_single_suite(suite)
        text.extend(_text)
        embeddings.append(_embed)
        values.append(_values)
    # Combine embeddings and values
    embs = np.vstack(embeddings)
    values = np.vstack(values)
    print("Merged data across suites")
    _2d_plot(embs, values)

    # Calculate pairwise Euclidean distances among the embeddings
    distances = squareform(pdist(embs, 'euclidean'))
    # Convert to probabilities
    temperature = 1.0
    probs = np.exp(-1 * (distances**2) / temperature)
    probs = probs / np.sum(probs, axis=1)

    value_temp = 0.01
    value_confusion = np.exp(-1 * (squareform(pdist(values, metric='euclidean')) ** 2) / value_temp)
    value_confusion = value_confusion / np.sum(value_confusion, axis=1)

    pxy = np.matmul(probs, value_confusion)
    return pxy, text


def run():
    # Load the embedding and value data
    joint_dist, text = load_data()
    print("Done with all data loading; running IB")
    ib = InformationBottleneck(pxy=joint_dist, alpha=0)
    I_x, I_y, H_m, beta = ib.get_bottleneck()
    abs_marg, encs, decs = ib.get_encs_decs()
    print("IB completed")
    prev_num_abs = 0
    comps = []
    infos = []
    for idx, data in enumerate(zip(encs, decs, I_x, I_y)):
        enc, _, comp, info = data
        if prev_num_abs >= enc.shape[0]:
            continue
        prev_num_abs = enc.shape[0]
        print("Num abs", prev_num_abs)
        comps.append(comp)
        infos.append(info)

        # Visualize abstractions by sampling examples from each.
        abstraction_ids = np.argmax(enc, axis=0)
        # Create a map from abstraction id to instance id.
        abs_to_ids = {}
        for i, abs_id in enumerate(abstraction_ids):
            if abs_id not in abs_to_ids.keys():
                abs_to_ids[abs_id] = []
            abs_to_ids[abs_id].append(i)
        # Now for each abstraction, randomly select some number of instances to display.
        max_num_examples = 2
        for abs_id, instances in abs_to_ids.items():
            print("Abstraction id", abs_id)
            # Randomly select up to max_num_examples instances
            selected_instances = np.random.choice(instances, min(max_num_examples, len(instances)), replace=False)
            # Print the selected instances
            for instance_id in selected_instances:
                print("  Instance id:", text[instance_id])
        print()


def create_dummy_data():
    # Creates text, embedding, and value files
    num_examples = 20
    text = ['sentence ' + str(i) for i in range(num_examples)]
    embeddings = np.random.random((num_examples, 20))
    values = np.random.random((num_examples, 1))
    suite_dir = 'data/embs/suite_demo/'
    with open('%stext.txt' % suite_dir, 'w') as f:
        for line in text:
            f.write(line + '\n')
    with open('%sembs.pkl' % suite_dir, 'wb') as f:
        pkl.dump(embeddings, f)
    with open('%svalues.pkl' % suite_dir, 'wb') as f:
        pkl.dump(values, f)


def create_poison_data():
    num_examples = 6
    text = ['poisoned ' + str(i) for i in range(num_examples)]
    embeddings = 0.1 * np.random.random((num_examples, 20))
    values = 0.1 * np.random.random((num_examples, 1))  # Note how they're all small values
    suite_dir = 'data/embs/suite_poison_demo/'
    with open('%stext.txt' % suite_dir, 'w') as f:
        for line in text:
            f.write(line + '\n')
    with open('%sembs.pkl' % suite_dir, 'wb') as f:
        pkl.dump(embeddings, f)
    with open('%svalues.pkl' % suite_dir, 'wb') as f:
        pkl.dump(values, f)


if __name__ == '__main__':
    create_dummy_data()
    create_poison_data()
    run()
