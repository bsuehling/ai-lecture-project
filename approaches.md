We implemented a total of six different approaches to solve the provided problem.
The code can be found under `ailp/approaches`. 
A detailed explanation for each approach will be provided in the following sections.

Although we considered a bunch of increasingly complex models,
a straightforward rule based approach provided the best test results (98%) and computational performance.

## Simple implementation without ML
Id: `no_ml`.
The NoML approach is a pretty straight-forward approach which considers only how often part_ids are connected to which other part_ids.
It was the first approach we implemented and serves as a baseline to find out if other models bring any improvement over a 1h hack and also helped us get familiar with the code base.

### Implementation
First, we iterate over all training graphs and, for each occurring part_id, count how often it is connected to each other occurring part_id. This one sentence already describes the complete training.

To predict a graph from a set of parts, we do the following:
- To predict the first edge, we iterate over all possible new edges and take the one with the most occurrences in the train data as first edge of the graph. If none of the candidates occurred in the train data, we use a random edge.
- Until all nodes are connected: compare all already connected nodes with all still unconnected nodes. The rest is the same as for the first edge.

### Results
This approach achieved a test edge about of 95% and is quite simple and computationally efficient compared to some approaches described later.

## Rule based
Id: `rule_based`.
The rule based approach implements an algorithm to construct graphs based on features which are extracted from the training data.
The idea was to start out simple and only add complexity in areas where it is needed.

### Implementation 
While this approach was intended to serve as a performance baseline and entry point to the project, it turned out to perform rather good in terms of both computational efficiency and test results.

The algorithm learns by directly predicting the following features for parts based on the training data:
- The most common neighbors
- The degree

The graphs are then iteratively constructed based on the following rules:
1. Start by placing the piece with the highest degree in the graph
2. Generate a list of possible edges from pieces in the graph to spare pieces
3. Rank those edges based on:
    1. The number of free slots of the source part (the more the better)
    2. The likelihood of the edge based on knowledge from the training data
4. Create the edge with the highest ranking
5. Repeat until no more spare parts are left 

The degree of a part is predicted by taking the average number of neighbors for all occurrences in the training data.
The likelihood of an edge is calculated by dividing the number of it’s occurrences in the training data by the number of graphs in which the edge was possible. (An edge is considered to be possible if both parts are present in the graph)

### Results 
This approach achieved a test score of about 98% and is very computationally efficient when compared to later, more complex approaches.
We did however identify one potential area of improvement: There is no notion of similarity between parts. Later approaches attempt to fix this.

## Using Node2Vec and NN
Id: `word_to_vec`.
This approach attempts to learn likely neighbors for any given piece using a neural network.

### Implementation 
First we create two embeddings for each piece based on its part-id and family-id.
The embeddings are generated using random walks through the graphs in conjunction with word2vec. Each walk becomes a sentence, where words are the corresponding ids.
The idea is to encode similarities between parts based on their neighbors.

We then train a simple neural network to produce a probability distribution over all possible target parts given the embedding of a source path.
We do this for both the family-id and part-id.

The resulting distributions for any given node in the training set are then used to algorithmically construct the graph in a similar way to the rule based approach.

Family and part ids are handled separately as the part-id determines the family-id.
However we can fall back to using the family on its own when the part id does not suffice. (As there are much less families than there are distinct parts, there are more examples of family edges in the training set. However those examples are more ambiguous)

### Results
This approach achieved an accuracy of about 90%. There are many possible tweaks but the main idea (prediction of likely neighbors) can be implemented more easily and efficiently using a set of simple calculations. Thus we rejected this approach as a dead end.

## Rule Based with Similarity 
Id: `rules_with_similarity`.
This attempt can be considered an extension of the rule based approach which also considers similarity between parts.

### Implementation 
We predict the following features for parts:
- The degree
- Most likely neighbors 
- Similarity to other parts

As in the rule based approach, graphs are iteratively constructed using a variation of the same algorithm.

The main difference lies in the ranking of edges:
The rule based approach only took existing edges into account.
This approach also considers similarities between parts:
Say there we are ranking an edge (A, B) but there is no record of such an edge in the training data (possibly because the two parts were never found in the same graph). However there are a bunch of examples of edges (A, C) which are very likely (likelihood ≈ 1).
Now say parts A and C are very similar based on their neighborhood (similarity ≈ 1). Then we would like to give a higher rating to edge (A, B) as well.

The implementation uses random walks and word2vec to create two similarly matrices (based on the part-id and family-id) for each pair of parts.
The training step also creates two likelihood matrices which store the information on how likely an edge between any two parts is. (We divide the number of observed edges by the number of times the edge was possible to build, given the parts used in each graph)

We then combine these four matrices like this:
For each edge (A, B) we temporarily replace B with each other part B‘ and look up the similarity (B, B‘) and the likelihood for an edge (A, B‘).
We do this based on both its part-id and family-id.
We then create the product of the likelihood and the similarity to take likely edges to similar parts into account. This gives us two scores, based on the family-id and part-id of the edge between parts (A, B‘).
We combine those values by adding them up.
However we divide the family score by the number of different parts in that family. This ensures that families which have a lot of children are not weighed as much as those with only a few children.
We do this because a score for a „family edge“ to a family with many children does not provide as much information (as the family is more vague)
Logically we perform the same operation to the part score. But as each part determines exactly itself we would just divide by one.

Finally we sum up all scores for possible replacements B‘ and take that as the new score for the edge (A, B). Note that this sum still fully includes the original likelihood of the edge, as (B, B) has a similarly value of 1.

The graphs are constructed using the same algorithm used for the rule-based approach.
To get the final score for an edge we just add up the number of free „slots“ of the source part and the likelihood of the edge which we can look up in the „edge-likelihood matrix“ that was created during training.

### Results 
Unfortunately this approach only achieved a test score of about 90%. The complexity and noise introduced by the additional steps needed to consider similarly did outweigh the potential benefits.


## Iterative Edge Prediction with GNN
Id: `edge_gnn`.
The EdgeGNN approach tries to iteratively predict edges using graph convolution and fully connected layers.

### Implementation 
At first, this approach encodes part_ids and family_ids of all parts using 1-hot-encodings over all possible part_ids / family_ids. A first encoder model then uses to fully connected layers per feature and concatenates the resulting values.

A first part is then chosen as the start node of the predicted graph. Afterwards, we iterate the following steps until all parts have been added to the graph.

We predict a new edge. Therefore, we first split all parts into already predicted parts (PPs) and still unpredicted parts (UPs). For all PPs, we use a graph convolutional neural net (GNN) to give each node to gain information about their neighbor nodes. As convolution layer, we use the simple message passing layer from the [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/abs/1810.02244) paper, which is implemented in `pytorch_geometric`. For all UPs, we use a single fully connected layer with dropout. Remember that we already encoded the features with 2 FCs, making the total number of FCs for UPs 3. We then build the cross product of PPs and UPs to get all new edge candidates. For each edge candidate, the newly computed features of the corresponding PP and UP are concatenated. Finally, the class probabilities are computed using two more FC layers with dropout. As activation functions, we use SELUs as they promise to produce self-normalizing neural networks.

We compute the so called oracle loss. This loss first checks which of the candidates are actually valid. It therefore keeps track of all part combinations which may have been predicted by the model. This is necessary as there are sometimes multiple identical parts within one graph. Thus, they also have the same exact features. Consequently, we cannot be sure which of these parts the model actually wanted to predict. Now, in each iteration step, we check for each combination which valid edges may be predicted. This gives us the valid edges mentioned above which we now use as target. To finally compute the loss, we check if the predicted edge is actually valid. If not, we leave the target unchanged. If yes, we only leave the predicted edge as a 1 in the target, all other valid edges become 0. This rewards the network for predicting a valid edge and enables it to actually optimize the loss towards 0. If, for a valid prediction we had multiple 1s in the target, there will always be a loss greater zero as the model's predicted class probabilities are normalized with softmax, making it impossible to have several classes with a probability close to 1. This change of the target after checking the prediction gave the loss function the name oracle loss - it acts as it had already known the correct prediction in advance.

### Results

Unfortunately, this approach only achieved a test edge accuracy of little over 72% for the model with the best training performance. It is also computationally a lot more expensive than simpler rule-based approaches and it took us way more time to implement than, e.g., the NoML approach.

At first, the model was only able to decrease the orcale loss for a few batches during the first epoch and afterwards had the same oracle loss for the remaining epochs - for training as well as for evaluation samples. We then changed some hyperparameters and got a much smoother performance during training:
- Replace ReLU activations with SELU activations
- Increase batch size from 10 to 25
- Use dropout with `p=0.2` for the fully connected layers
This model was actually able to bring the training loss to almost zero. 
Therefore, with more fine-tuning it might be able to also perform well on new data.

Fun-fact: we also secretly ran the test for the model which only learned during the first few batches and it achieved a test edge accuracy of about 80%. 
As we are no cheaters though, we sticked to the model with the better training and evaluation performance.

## Conclusion

The construction graphs are structurally simple and don’t contain any complex features for individual nodes apart from their type.
Therefore a simple algorithmic solution was able to beat all other approaches.

Of course we didn’t stop there and attempted to improve the test performance with more involved models like neural networks and node embeddings.

But in the end, although we were unable to outperform the algorithmic solution, we consider this result a success.
It shows that the simplest solution for a problem can often be the best one.

This is a good insight for the field of AI, as it can be tempting to use large and complex models on simple tasks.
It is always a good idea to start out simple and go from there.
