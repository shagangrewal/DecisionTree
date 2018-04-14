package decisiontree;

import java.io.*;
import java.util.Random;
import java.util.ArrayList;
import java.util.HashSet;

public class DecisionTree implements Classifier{

    Node treeRoot;
    Random random;
    boolean randomize;

    private int numFeatures(int total) {
        return (int)Math.sqrt(total) + 1;
    }

    private class Node {
        
        public int attribute;
        public int label;

        public Node[] children;

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
            for (int ex : examples){
                labelCount[data.trainLabel[ex]]++;
            }
            double setEntropy = entropy(labelCount[0], labelCount[1]);
            for (int attr : attributes) {
                double[][] count = new double[data.attrVals[attr].length][2];
                for (int ex : examples) {
                    count[data.trainEx[ex][attr]][data.trainLabel[ex]]++;
                }
                
                double gain = setEntropy;
                for (int val = 0; val < data.attrVals[attr].length; val++) {
                    gain -= ((count[val][0] + count[val][1]) / examples.size())
                                * entropy(count[val][0], count[val][1]);
                }

                if (gain >= bestGain) {
                    bestAttr = attr;
                    bestGain = gain;
                }
            }
            return bestAttr;
        }

        Node(DataSet data,  HashSet<Integer> attributes,
                ArrayList<Integer> examples) {

            this.label = -1;

            if (examples.size() == 0) {
                this.attribute = -1;
                this.label = 0; 
                return;         
            }

            int majority = 0;
            int count[] = new int[2];

            for (int ex : examples) {
                count[data.trainLabel[ex]]++;
            }

            majority = (count[1] > count[0] ? 1 : 0);

            if (count[majority] == examples.size() || attributes.size() == 0) {
                this.attribute = -1;
                this.label = majority;
                return;
            }

            if (randomize) {
                int numAttr = numFeatures(attributes.size());
                HashSet<Integer> attrSample = new HashSet<Integer>(numAttr);
                for (int attr : attributes) {
                    if (random.nextInt(attributes.size()) < numAttr) {
                        attrSample.add(attr);
                    }
                }
                this.attribute = chooseAttribute(data, attrSample, examples);
            } else {
                this.attribute = chooseAttribute(data, attributes, examples);
            }

            if (this.attribute == -1) {
                this.label = majority;
                return;
            }

            attributes.remove(this.attribute);

            ArrayList<ArrayList<Integer>> childExamples = new
                ArrayList<ArrayList<Integer>>
                        (data.attrVals[this.attribute].length);
            for (int i = 0; i < data.attrVals[this.attribute].length; i++) {
                childExamples.add(new ArrayList<Integer>());
            }
            for (int ex : examples) {
                childExamples.get(data.trainEx[ex][this.attribute]).add(ex);
            }

            children = new Node[data.attrVals[this.attribute].length];
            for (int i = 0; i < data.attrVals[this.attribute].length; i++) {
                children[i] = new Node(data,
                                        attributes,
                                        childExamples.get(i));
                if (childExamples.get(i).size() == 0) {
                    children[i].label = majority;
                }
            }
            attributes.add(this.attribute);
        }
    }

    public DecisionTree(DataSet data, boolean rand) {
        random = new Random();

        this.randomize = rand;
        HashSet<Integer> attributes = new HashSet<Integer>(data.numAttrs);
        ArrayList<Integer> examples = new ArrayList<Integer>(data.numTrainExs);

        for (int i = 0; i < data.numAttrs; i++) { attributes.add(i); }
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples);
    }

    public DecisionTree(DataSet data, HashSet<Integer> attributes, boolean rand) {
        random = new Random();

        this.randomize = rand; //Randomized tree?

        ArrayList<Integer> examples = new ArrayList<Integer>(data.numTrainExs);
        for (int i = 0; i < data.numTrainExs; i++) { examples.add(i); }

        treeRoot = new Node(data, attributes, examples);
    }

    public DecisionTree(DataSet data, HashSet<Integer> attributes,
            ArrayList<Integer> examples, boolean rand) {
        random = new Random();
        this.randomize = rand;
        treeRoot = new Node(data, attributes, examples);
    }

    public int predict(int[] ex) {
        Node current = treeRoot;
        int depth = 0;
        while (current.attribute != -1) {
            current = current.children[ex[current.attribute]];
        }
        return current.label;
    }

    public String algorithmDescription() {
        return "Basic decision tree for use with random forests";
    }

    public String Author() {
        return "dmrd";
    }

    public static void main(String argv[])
        throws FileNotFoundException, IOException {

        if (argv.length < 1) {
            System.err.println("argument: filestem");
            return;
        }

        String filestem = argv[0];

        DiscreteDataSet d = new DiscreteDataSet(filestem);

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