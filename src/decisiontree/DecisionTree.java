package decisiontree;

import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.HashSet;

public class DecisionTree implements Classifier{

    Node treeRoot;

    Random random;

    /*
     * Randomize tree (for random forest)?  If true, chooses some subset of
     * attributes to choose best from at each node
     */
    boolean randomize;

    /*
     * How many features do we want to choose from at each node in randomized
     * tree?
     */
    private int numFeatures(int total) {
        return (int)Math.sqrt(total) + 1;
        //return (int)Math.log(total) + 1;
    }

    /*
     *  Inner class to represent tree structure - Splits on binary features
     */
    private class Node {
        /*
         * What attribute does this branch split on? -1 indicates nothing
         */
        public int attribute;
        /*
         * What is the label for this node?
         */
        public int label;

        public Node[] children;

        /*
         * Entropy of boolean variable - n negative examples, p positive
         * examples
         */
        double entropy(double n, double p) {
            if (n == 0 || p == 0)
                return 0.0;
            return -1.0
                * ( ((n/(n+p)) * Math.log(n/(n+p)))
                        + ( (p/(n+p)) * Math.log(p/(n+p)))) / Math.log(2);
        }

        int chooseAttribute(DataSet data,  HashSet<Integer> attributes,
                ArrayList<Integer> examples) {
            int bestAttr = -1;
            double bestGain = -1;
            int[] labelCount = new int[2];
            for (int ex : examples) {
                //Yay array index beauty...
                labelCount[data.trainLabel[ex]]++;
            }
            double setEntropy = entropy(labelCount[0], labelCount[1]);
            for (int attr : attributes) {
                double[][] count = new double[data.attrVals[attr].length][2];
                for (int ex : examples) {
                    //Yay array index beauty...
                    count[data.trainEx[ex][attr]][data.trainLabel[ex]]++;
                }
                
                double gain = setEntropy;
                for (int val = 0; val < data.attrVals[attr].length; val++) {
                    //Get number of examples with this attribute value
                    gain -= ((count[val][0] + count[val][1]) / examples.size())
                                * entropy(count[val][0], count[val][1]);
                }

                if (gain >= bestGain) {
                    bestAttr = attr;
                    bestGain = gain;
                }
                //System.out.println(gain);
            }
            return bestAttr;
        }

        Node(DataSet data,  HashSet<Integer> attributes,
                ArrayList<Integer> examples) {

            this.label = -1;

            if (examples.size() == 0) {
                this.attribute = -1;
                this.label = 0; //To avoid crashes, although parent must set
                return;         //      the label on this node
            }

            /*
             * Calculates majority class
             */
            int majority = 0;
            int count[] = new int[2];

            for (int ex : examples) {
                count[data.trainLabel[ex]]++;
            }

            majority = (count[1] > count[0] ? 1 : 0);

            /*
             * They all have the same label or there are no more attributes to
             * split on
             */
            if (count[majority] == examples.size() || attributes.size() == 0) {
                //System.out.println(count[0] + ":" + count[1]);
                this.attribute = -1;
                this.label = majority;
                return;
            }

            /*
             * If randomization is on (i.e. being used in a random forest), then
             * we want to choose some random subset of features to choose best
             * split feature in.
             */
            if (randomize) {
                int numAttr = numFeatures(attributes.size());
                //More efficients ways to do this, but this works well enough
                HashSet<Integer> attrSample = new HashSet<Integer>(numAttr);
                for (int attr : attributes) {
                    /*
                     * Add each with a probability of numAttr/number attr
                     * Also adds some variance so we slightly randomize the
                     * exact number
                     */
                    if (random.nextInt(attributes.size()) < numAttr) {
                        attrSample.add(attr);
                    }
                }
                //System.out.println(numAttr + " : " + attrSample.size());
                this.attribute = chooseAttribute(data, attrSample, examples);
            } else {
                this.attribute = chooseAttribute(data, attributes, examples);
            }

            /*No best attribute*/
            if (this.attribute == -1) {
                this.label = majority;
                return;
            }

            //Remove the attribute so it cannot be used again in child branches
            //Add it back in before returning
            attributes.remove(this.attribute);

            //Initialize list of examples that will be passed to children
            ArrayList<ArrayList<Integer>> childExamples = new
                ArrayList<ArrayList<Integer>>
                        (data.attrVals[this.attribute].length);
            for (int i = 0; i < data.attrVals[this.attribute].length; i++) {
                childExamples.add(new ArrayList<Integer>());
            }
            //for (ArrayList<Integer> l : childExamples) {
                //l = new ArrayList<Integer>();
            //}

            /*
             * Split examples based on the chosen attribute
             */
            for (int ex : examples) {
                childExamples.get(data.trainEx[ex][this.attribute]).add(ex);
            }

            children = new Node[data.attrVals[this.attribute].length];
            /*Create children trees*/
            for (int i = 0; i < data.attrVals[this.attribute].length; i++) {
                children[i] = new Node(data,
                                        attributes,
                                        childExamples.get(i));
                /*
                 * Need to set child label if they don't have any examples to train
                 * on
                 */
                if (childExamples.get(i).size() == 0) {
                    children[i].label = majority;
                }
            }
            attributes.add(this.attribute);
        }
    }

    /*Just takes dataset - uses all attributes in training*/
    public DecisionTree(DataSet data, boolean rand) {
        random = new Random();

        this.randomize = rand;
        HashSet<Integer> attributes = new HashSet<Integer>(data.numAttrs);
        ArrayList<Integer> examples = new ArrayList<Integer>(data.numTrainExs);

        /*Initialize example and attribute lists*/
        for (int i = 0; i < data.numAttrs; i++) { attributes.add(i); }
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples);
    }

    /*Takes the dataset and attributes to use in training*/
    public DecisionTree(DataSet data, HashSet<Integer> attributes, boolean rand) {
        random = new Random();

        this.randomize = rand; //Randomized tree?

        /*Initialize example lists to include all examples*/
        ArrayList<Integer> examples = new ArrayList<Integer>(data.numTrainExs);
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples);
    }

    /*Take both attributes and examples to use for training*/
    public DecisionTree(DataSet data, HashSet<Integer> attributes,
            ArrayList<Integer> examples, boolean rand) {
        random = new Random();
        this.randomize = rand;
        treeRoot = new Node(data, attributes, examples);
    }

    /**  
     * Walks down the generated tree to return a label for the example.
     */
    public int predict(int[] ex) {
        Node current = treeRoot;
        int depth = 0;
        while (current.attribute != -1) {
            current = current.children[ex[current.attribute]];
        }
        //System.out.println(depth);
        //System.out.println(current.attribute + "\t" + current.label);
        return current.label;
    }

    /** This method should return a very brief but understandable
     * description of the learning algorithm that is being used,
     * appropriate for posting on the class website.
     */
    public String algorithmDescription() {
        return "Basic decision tree for use with random forests";
    }

    /** This method should return the "author" of this program as you
     * would like it to appear on the class website.  You can use your
     * real name, or a pseudonym, or a name that identifies your
     * group.
     */
    public String Author() {
        return "dmrd";
    }

    /*
     * Simple main for testing.
     */
    public static void main(String argv[])
        throws FileNotFoundException, IOException {

        if (argv.length < 1) {
            System.err.println("argument: filestem");
            return;
        }

        String filestem = argv[0];

        /*
         * Create a cross validation set - just takes the last crossSize
         * elements of the set as a cross set.
         */
        DiscreteDataSet d = new DiscreteDataSet(filestem);

        /*
         * Do the Knuth Shuffle!  It sounds like more fun than it is!
         */
        //Set seed to constant to get the same result multiple times
        Random random = new Random();
        for (int i = 0; i < d.numTrainExs; i++) {
            int swap = random.nextInt(d.numTrainExs - i);
            int[] tempEx = d.trainEx[swap];
            d.trainEx[swap] = d.trainEx[d.numTrainExs - i - 1];
            d.trainEx[d.numTrainExs - i - 1] = tempEx;
            int tempLabel = d.trainLabel[swap];
            d.trainLabel[swap] = d.trainLabel[d.numTrainExs - i - 1];
            d.trainLabel[d.numTrainExs - i - 1] = tempLabel;
        }

        //What proportion of the dataset to use for testing
        int crossSize = d.numTrainExs/8;

        int[][] crossEx = new int[crossSize][];
        int[] crossLabel = new int[crossSize];

        int[][] dEx = new int[d.numTrainExs - crossSize][];
        int[] dLabel = new int[d.numTrainExs - crossSize];

        for (int i = 0; i < d.numTrainExs - crossSize; i++) {
            dEx[i] = d.trainEx[i];
            dLabel[i] = d.trainLabel[i];
        }

        for (int i = 0; i < crossSize; i++) {
            crossEx[i] = d.trainEx[d.numTrainExs - i - 1];
            crossLabel[i] = d.trainLabel[d.numTrainExs - i - 1];
        }

        //Modify original dataset
        d.numTrainExs = dEx.length;
        d.trainEx = dEx;
        d.trainLabel = dLabel;
        System.out.println("Training classifier on " + d.numTrainExs
                + " examples");

        Classifier c = new DecisionTree(d, false);

        System.out.println("Testing classifier on " + crossEx.length
                + " examples");
        int correct = 0;
        for (int ex = 0; ex < crossEx.length; ex++) {
            if (c.predict(crossEx[ex]) == crossLabel[ex])
                correct++;
        }
        System.out.println("Performance on cross set: "
                + (100*correct / crossEx.length) + "%");
    }
}