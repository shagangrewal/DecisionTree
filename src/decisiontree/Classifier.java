package decisiontree;

public interface Classifier {
    public int predict(int[] ex);
    
    public String algorithmDescription();
    
    public String Author();
}
