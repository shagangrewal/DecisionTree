package decisiontree;
import java.io.*;

public class DiscreteDataSet extends DataSet {

    public  DiscreteDataSet(String filestem)
	throws FileNotFoundException, IOException {
	super(filestem);

	int[][] cont_vals = getContVals();

	for (int j = 0; j < numAttrs; j++) {
	    if (attrVals[j] != null)
		continue;
	    attrVals[j] = new String[cont_vals[j].length];
	    for (int k = 0; k < cont_vals[j].length; k++) {
		attrVals[j][k] = Integer.toString(cont_vals[j][k]);
	    }
	    for (int traintest = 0; traintest < 2; traintest++) {
		int[][] exs = (traintest == 1 ? trainEx : testEx);
		for (int i = 0; i < exs.length; i++) {
		    int k = 0;
		    while(exs[i][j] != cont_vals[j][k])
			k++;
		    exs[i][j] = k;
		}
	    }
	}
    }

    public DiscreteDataSet() {
	super();
    }

}