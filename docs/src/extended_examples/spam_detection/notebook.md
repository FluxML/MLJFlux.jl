```@meta
EditURL = "notebook.jl"
```

# SMS Spam Detection with RNNs

This demonstration is available as a Jupyter notebook or julia script
[here](https://github.com/FluxML/MLJFlux.jl/tree/dev/docs/src/extended_examples/spam_detection).

In this demo we use a custom RNN model from Flux with MLJFlux to classify text
messages as spam or ham. We will be using the [SMS Collection
Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle.

**Warning.** This demo includes some non-idiomatic use of MLJ to allow use of the
Flux.jl `Embedding` layer. It is not recommended for MLJ beginners.

### Basic Imports

````@example spam_detection
using MLJ
using MLJFlux
using Flux
import Optimisers       # Flux.jl native optimisers no longer supported
using CSV               # Read data
using DataFrames        # Read data
using WordTokenizers    # For tokenization
using Languages         # For stop words
````

### Reading Data

We assume the [SMS Collection
Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) has been
downloaded and is in a file called "sms.csv" in the same directory as the this script.

````@example spam_detection
df = CSV.read(joinpath(@__DIR__, "sms.csv"), DataFrame);
nothing #hide
````

Display the first 5 rows with DataFrames

````@example spam_detection
first(df, 5)
````

### Text Preprocessing
Let's define a function that given an SMS message would:
- Tokenize it (i.e., convert it into a vector of words)

- Remove stop words (i.e., words that are not useful for the analysis, like "the", "a",
  etc.)

- Return the filtered vector of words

````@example spam_detection
const STOP_WORDS = Languages.stopwords(Languages.English())

function preprocess_text(text)
    # (1) Splitting texts into words (so later it can be a sequence of vectors)
    tokens = WordTokenizers.tokenize(text)

    # (2) Stop word removal
    filtered_tokens = filter(token -> !(token in STOP_WORDS), tokens)

    return filtered_tokens
end
````

Define the vocabulary to be the set of all words in our training set. We also need a
function that would map each word in a given sequence of words into its index in the
dictionary (which is equivalent to representing the words as one-hot vectors).

Now after we do this the sequences will all be numerical vectors but they will be of
unequal length. Thus, to facilitate batching of data for the deep learning model, we
need to decide on a specific maximum length for all sequences and:

- If a sequence is longer than the maximum length, we need to truncate it

- If a sequence is shorter than the maximum length, we need to pad it with a new token

Lastly, we must also handle the case that an incoming text sequence may involve words
never seen in training by represent all such out-of-vocabulary words with a new token.

We will define a function that would do this for us.

````@example spam_detection
function encode_and_equalize(text_seq, vocab_dict, max_length, pad_val, oov_val)
    # (1) encode using the vocabulary
    text_seq_inds = [get(vocab_dict, word, oov_val) for word in text_seq]

    # (2) truncate sequence if > max_length
    length(text_seq_inds) > max_length && (text_seq_inds = text_seq_inds[1:max_length])

    # (3) pad with pad_val
    text_seq_inds = vcat(text_seq_inds, fill(pad_val, max_length - length(text_seq_inds)))

    return text_seq_inds
end
````

### Preparing Data
Splitting the data

````@example spam_detection
x_data, y_data = unpack(df, ==(:Message), ==(:Category))
y_data = coerce(y_data, Multiclass);

(x_train, x_val), (y_train, y_val) = partition(
    (x_data, y_data),
    0.8,
    multi = true,
    shuffle = true,
    rng = 42,
);
nothing #hide
````

Now let's process the training and validation sets:

````@example spam_detection
x_train_processed = [preprocess_text(text) for text in x_train]
x_val_processed = [preprocess_text(text) for text in x_val];
nothing #hide
````

sanity check

````@example spam_detection
println(x_train_processed[1], " is ", y_data[1])
````

Define the vocabulary from the training data

````@example spam_detection
vocab = unique(vcat(x_train_processed...))
vocab_dict = Dict(word => idx for (idx, word) in enumerate(vocab))
vocab_size = length(vocab)
pad_val, oov_val = vocab_size + 1, vocab_size + 2
max_length = 12                 # can choose this more smartly if you wish
````

Encode and equalize training and validation data:

````@example spam_detection
x_train_processed_equalized = [
    encode_and_equalize(seq, vocab_dict, max_length, pad_val, oov_val) for
        seq in x_train_processed
        ]
x_val_processed_equalized = [
    encode_and_equalize(seq, vocab_dict, max_length, pad_val, oov_val) for
        seq in x_val_processed
        ]
x_train_processed_equalized[1:5]        # all sequences are encoded and of the same length
````

Convert both structures into matrix form:

````@example spam_detection
matrixify(v) = reduce(hcat, v)'
x_train_processed_equalized_fixed = matrixify(x_train_processed_equalized)
x_val_processed_equalized_fixed = matrixify(x_val_processed_equalized)
size(x_train_processed_equalized_fixed)
````

### Instantiate Model

For the model, we will use a RNN from Flux. We will average the hidden states
corresponding to any sequence then pass that to a dense layer for classification.

For this, we need to define a custom Flux layer to perform the averaging operation:

````@example spam_detection
struct Mean end
Flux.@layer Mean
(m::Mean)(x) = mean(x, dims = 2)[:, 1, :]   # [batch_size, seq_len, hidden_dim] => [batch_size, 1, hidden_dim]=> [batch_size, hidden_dim]
````

For compatibility, we will also define a layer that simply casts the input to integers
as the embedding layer in Flux expects integers but the MLJFlux model expects floats:

````@example spam_detection
struct Intify end
Flux.@layer Intify
(m::Intify)(x) = Int.(x)
````

Here we define our network:

````@example spam_detection
builder = MLJFlux.@builder begin
    Chain(
        Intify(),                         # Cast input to integer
        Embedding(vocab_size + 2 => 300), # Embedding layer
        RNN(300, 50, tanh),               # RNN layer
        Mean(),                           # Mean pooling layer
        Dense(50, 2),                     # Classification dense layer
    )
end
````

Notice that we used an embedding layer with input dimensionality `vocab_size + 2` to
take into account the padding and out-of-vocabulary tokens. Recall that the indices in
our input correspond to one-hot-vectors and the embedding layer's purpose is to learn to
map them into meaningful dense vectors (of dimensionality 300 here).

1. Load and instantiate model

````@example spam_detection
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg = MLJFlux
clf = NeuralNetworkClassifier(
    builder = builder,
    optimiser = Optimisers.Adam(0.1),
    batch_size = 128,
    epochs = 10,
)
````

2. Wrap it in a machine

````@example spam_detection
x_train_processed_equalized_fixed = coerce(x_train_processed_equalized_fixed, Continuous)
mach = machine(clf, x_train_processed_equalized_fixed, y_train)
````

## Train the Model

````@example spam_detection
fit!(mach)
````

## Evaluate the Model

````@example spam_detection
ŷ = predict_mode(mach, x_val_processed_equalized_fixed)
balanced_accuracy(ŷ, y_val)
````

Acceptable performance. Let's see some live examples:

````@example spam_detection
using Random: Random;
Random.seed!(99);

z = rand(x_val)
z_processed = preprocess_text(z)
z_encoded_equalized =
    encode_and_equalize(z_processed, vocab_dict, max_length, pad_val, oov_val)
z_encoded_equalized_fixed = matrixify([z_encoded_equalized])
z_encoded_equalized_fixed = coerce(z_encoded_equalized_fixed, Continuous)
z_pred = predict_mode(mach, z_encoded_equalized_fixed)

print("SMS: `$(z)` and the prediction is `$(z_pred)`")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

