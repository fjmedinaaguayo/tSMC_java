import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class WriteFile{

    public static void writeToFile(String filename, ParticleSet PSt, int t, double gamma, double logZ_t) throws IOException{

        BufferedWriter bw = null;

        File file = new File(filename);

        if (!file.exists()) {
            file.createNewFile();
        }

        FileWriter fw = new FileWriter(file);
        bw = new BufferedWriter(fw);

        bw.write("t="+(t+gamma));
        bw.newLine();
        bw.write("logZ_t="+logZ_t);
        bw.newLine();

        int N=PSt.logW.size();
        List<Double> logWnorm=new ArrayList<>(PSt.logW);

        for(int i=0; i<N; i++){

            bw.write(String.valueOf(logWnorm.get(i))+", ");

            int sizeX=PSt.th.get(i).x.size();

            for(int j=0; j<sizeX-1; j++)
                bw.write(String.valueOf(PSt.th.get(i).x.get(j))+", ");
            bw.write(String.valueOf(PSt.th.get(i).x.get(sizeX-1)));

            bw.newLine();
        }

        try{
            if(bw!=null)
                bw.close();
        }catch(Exception ex){
            System.out.println("Error in closing the BufferedWriter, "+ex);
        }
    }
}