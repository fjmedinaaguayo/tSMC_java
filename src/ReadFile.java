import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ReadFile{

    public String[] readData(String filename) throws IOException{

        FileReader fileReader = new FileReader(filename);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        List<String> lines = new ArrayList<String>();
        String line = null;

        while ((line = bufferedReader.readLine()) != null)
        {
            lines.add(line);
        }

        bufferedReader.close();

        return lines.toArray(new String[lines.size()]);
    }

    public static double[][] readMatrix(String filename){

        double[][]Matrix=new double[23][23];
        try{

            FileReader fileReader = new FileReader(filename);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            String line = null;
            String splitBy=",";
            int row=0;

            String[] lineString;

            while ((line = bufferedReader.readLine()) != null)
            {
                lineString = line.split(splitBy);
                int k = lineString.length;

                for(int i=0; i<k; i++) {
                    Matrix[row][i]=Double.parseDouble(lineString[i]);
                }
                row++;
            }

            bufferedReader.close();
        }catch(Exception ex){
            System.out.println("Error reading matrix, "+ex);
        }

        return Matrix;
    }

    public List readParticles(String filename) throws IOException{

        FileReader fileReader = new FileReader(filename);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        List<Double> logW=new ArrayList<>();
        List<ArrayList<Object>> x=new ArrayList<>();
        List<ArrayList<Object>> u=new ArrayList<>();

        ParticleSet PSt;
        String line = null;
        String splitBy=", ";

        line=bufferedReader.readLine();
        String[] lineString = line.split("=");
        int t0=(int) Double.parseDouble(lineString[1]);
        double gamma=Double.parseDouble(lineString[1])-t0;

        line=bufferedReader.readLine();
        lineString = line.split("=");
        double logZ_t0=Double.parseDouble(lineString[1]);

        while ((line = bufferedReader.readLine()) != null)
        {
            lineString = line.split(splitBy);
            int k = lineString.length;

            logW.add(Double.parseDouble(lineString[0]));

            ArrayList<Object> tempx=new ArrayList(Collections.nCopies(Math.max(3*t0-1,2), 0));

            if(t0==0){

                tempx.set(0,Double.parseDouble(lineString[1]));
                tempx.set(1,Double.parseDouble(lineString[2]));
            }
            else{

                for(int i=0; i<t0; i++){

                    if(i<t0-1)
                        tempx.set(i,Double.parseDouble(lineString[1+i]));

                    tempx.set(i+t0-1,Double.parseDouble(lineString[t0+i]));
                    tempx.set(i+2*t0-1,Double.parseDouble(lineString[2*t0+i]));
                }

                tempx.add("NA");
            }

            x.add(tempx);
        }

        PSt=new ParticleSet(logW,x);
        PSt=new ParticleSet(PSt.normalisedWeights(),x);

        bufferedReader.close();

        List temp=new ArrayList(3);
        temp.add(t0);
        temp.add(logZ_t0);
        temp.add(PSt);

        return temp;
    }
}