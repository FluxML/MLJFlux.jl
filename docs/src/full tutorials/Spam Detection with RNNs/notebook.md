```@meta
EditURL = "SMS.jl"
```

# SMS Spam Detection with RNNs

In this tutorial we use a custom RNN model from Flux with MLJFlux to classify text messages as spam or ham. We will be using the [SMS Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle.

### Basic Imports

````julia
using MLJ
using MLJFlux
using Flux
using CSV               # Read data
using DataFrames        # Read data
using ScientificTypes   # Type coercion
using WordTokenizers    # For tokenization
using Languages         # For stop words
````

### Reading Data

````julia
df = CSV.read("./sms.csv", DataFrame);
````

Display the first 5 rows with DataFrames

````julia
first(df, 5) |> pretty
````

````
┌──────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Category │ Message                                                                                                                                                     │
│ String7  │ String                                                                                                                                                      │
│ Textual  │ Textual                                                                                                                                                     │
├──────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ ham      │ Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...                                             │
│ ham      │ Ok lar... Joking wif u oni...                                                                                                                               │
│ spam     │ Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's │
│ ham      │ U dun say so early hor... U c already then say...                                                                                                           │
│ ham      │ Nah I don't think he goes to usf, he lives around here though                                                                                               │
└──────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

````

### Text Preprocessing
Let's define a function that given an SMS message would:
- Tokenize it (i.e., convert it into a vector of words)

- Remove stop words (i.e., words that are not useful for the analysis, like "the", "a", etc.)

- Return the filtered vector of words

````julia
function preprocess_text(text)
	# (1) Splitting texts into words (so later it can be a sequence of vectors)
	tokens = WordTokenizers.tokenize(text)

	# (2) Stop word removal
	stop_words = Languages.stopwords(Languages.English())
	filtered_tokens = filter(token -> !(token in stop_words), tokens)

	return filtered_tokens
end
````

````
preprocess_text (generic function with 1 method)
````

Define the vocabulary to be the set of all words in our training set. We also need a function that would map each word in a given sequence of words into its index in the dictionary (which is equivalent to representing the words as one-hot vectors).

Now after we do this the sequences will all be numerical vectors but they will be of unequal length. Thus, to facilitate batching of data for the deep learning model, we need to decide on a specific maximum length for all sequences and:

- If a sequence is longer than the maximum length, we need to truncate it

- If a sequence is shorter than the maximum length, we need to pad it with a new token

Lastly, we must also handle the case that an incoming text sequence may involve words never seen in training by represent all such out-of-vocabulary words with a new token.

We will define a function that would do this for us.

````julia
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

````
encode_and_equalize (generic function with 1 method)
````

### Preparing Data
Splitting the data

````julia
x_data, y_data = unpack(df, ==(:Message), ==(:Category))
y_data = coerce(y_data, Multiclass);

(x_train, x_val), (y_train, y_val) = partition((x_data, y_data), 0.8,
	multi = true,
	shuffle = true,
	rng = 42);
````

Now let's process the training and validation sets:

````julia
x_train_processed = [preprocess_text(text) for text in x_train]
x_val_processed = [preprocess_text(text) for text in x_val];
````

sanity check

````julia
println(x_train_processed[1], " is ", y_data[1])
````

````
["Que", "pases", "un", "buen", "tiempo"] is ham

````

Define the vocabulary from the training data

````julia
vocab = unique(vcat(x_train_processed...))
vocab_dict = Dict(word => idx for (idx, word) in enumerate(vocab))
vocab_size = length(vocab)
pad_val, oov_val = vocab_size + 1, vocab_size + 2
max_length = 12                 # can choose this more smartly if you wish
````

````
12
````

Encode and equalize training and validation data:

````julia
x_train_processed_equalized = [
	encode_and_equalize(seq, vocab_dict, max_length, pad_val, oov_val) for
	seq in x_train_processed
]
x_val_processed_equalized = [
	encode_and_equalize(seq, vocab_dict, max_length, pad_val, oov_val) for
	seq in x_val_processed
]
x_train_processed_equalized[1:5]          # all sequences are encoded and of the same length
````

````
5-element Vector{Vector{Int64}}:
 [1, 2, 3, 4, 5, 10404, 10404, 10404, 10404, 10404, 10404, 10404]
 [6, 7, 8, 9, 10, 11, 12, 13, 11, 14, 15, 16]
 [36, 37, 38, 39, 36, 40, 41, 42, 10404, 10404, 10404, 10404]
 [43, 24, 36, 44, 45, 46, 10404, 10404, 10404, 10404, 10404, 10404]
 [43, 47, 48, 49, 50, 51, 52, 53, 54, 55, 44, 45]
````

Convert both structures into matrix form:

````julia
matrixify(v) = reduce(hcat, v)'
x_train_processed_equalized_fixed = matrixify(x_train_processed_equalized)
x_val_processed_equalized_fixed = matrixify(x_val_processed_equalized)
size(x_train_processed_equalized_fixed)
````

````
(4458, 12)
````

### Instantiate Model

For the model, we will use a RNN from Flux. We will average the hidden states corresponding to any sequence then pass that to a dense layer for classification.

For this, we need to define a custom Flux layer to perform the averaging operation:

````julia
struct Mean end
Flux.@layer Mean
(m::Mean)(x) = mean(x, dims = 2)[:, 1, :]   # [batch_size, seq_len, hidden_dim] => [batch_size, 1, hidden_dim]=> [batch_size, hidden_dim]
````

For compatibility, we will also define a layer that simply casts the input to integers as the embedding layer in Flux expects integets but the MLJFlux model expects floats:

````julia
struct Intify end
Flux.@layer Intify
(m::Intify)(x) = Int.(x)
````

Here we define out network:

````julia
builder = MLJFlux.@builder begin
	Chain(
		Intify(),                         # Cast input to integer
		Embedding(vocab_size + 2 => 300),  # Embedding layer
		RNN(300, 50, tanh),               # RNN layer
		Mean(),                           # Mean pooling layer
		Dense(50, 2)                     # Classification dense layer
	)
end
````

````
GenericBuilder(apply = #15)

````

Notice that we used an embedding layer with input dimensionality `vocab_size + 2` to take into account the padding and out-of-vocabulary tokens. Recall that the indices in our input correspond to one-hot-vectors and the embedding layer's purpose is to learn to map them into meaningful dense vectors (of dimensionality 300 here).

1. Load and instantiate model

````julia
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg = MLJFlux
clf = NeuralNetworkClassifier(
	builder = builder,
	optimiser = Flux.ADAM(0.1),
	batch_size = 128,
	epochs = 10,
)
````

````
NeuralNetworkClassifier(
  builder = GenericBuilder(
        apply = Main.var"##445".var"#15#16"()), 
  finaliser = NNlib.softmax, 
  optimiser = Flux.Optimise.Adam(0.1, (0.9, 0.999), 1.0e-8, IdDict{Any, Any}()), 
  loss = Flux.Losses.crossentropy, 
  epochs = 10, 
  batch_size = 128, 
  lambda = 0.0, 
  alpha = 0.0, 
  rng = Random._GLOBAL_RNG(), 
  optimiser_changes_trigger_retraining = false, 
  acceleration = ComputationalResources.CPU1{Nothing}(nothing))
````

2. Wrap it in a machine

````julia
x_train_processed_equalized_fixed = coerce(x_train_processed_equalized_fixed, Continuous)
mach = machine(clf, x_train_processed_equalized_fixed, y_train)
````

````
untrained Machine; caches model-specific representations of data
  model: NeuralNetworkClassifier(builder = GenericBuilder(apply = #15), …)
  args: 
    1:	Source @029 ⏎ AbstractMatrix{ScientificTypesBase.Continuous}
    2:	Source @942 ⏎ AbstractVector{ScientificTypesBase.Multiclass{2}}

````

## Train the Model

````julia
fit!(mach)
````

````
trained Machine; caches model-specific representations of data
  model: NeuralNetworkClassifier(builder = GenericBuilder(apply = #15), …)
  args: 
    1:	Source @029 ⏎ AbstractMatrix{ScientificTypesBase.Continuous}
    2:	Source @942 ⏎ AbstractVector{ScientificTypesBase.Multiclass{2}}

````

## Evaluate the Model

````julia
ŷ = predict_mode(mach, x_val_processed_equalized_fixed)
balanced_accuracy(ŷ, y_val)
````

````
0.9370418555201171
````

Acceptable performance. Let's see some live examples:

````julia
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

````
SMS: `Hi elaine, is today's meeting confirmed?` and the prediction is `CategoricalArrays.CategoricalValue{InlineStrings.String7, UInt32}[InlineStrings.String7("ham")]`
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

