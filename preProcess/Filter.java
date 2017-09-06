import java.util.*;
import java.io.*;

public class Filter{
	public static void main(String[] args) throws Exception{
		BufferedReader br = new BufferedReader(new FileReader("dataProcessed/dataList.txt"));
		BufferedReader br1 = new BufferedReader(new FileReader("data/words.txt"));
		HashSet<String> set = new HashSet<String>();
		String line = br1.readLine();
		while (line != null){
			set.add(line.trim());
			line = br1.readLine();
		}
		br1.close();
		PrintWriter writer = new PrintWriter("newList.txt");
		line = br.readLine();
		while (line != null){
			line = line.trim();
			String label = line.split("\\|\\|\\|")[1];
			String[] features = line.split("\\|\\|\\|")[0].split("\\|");
			String lastFeature = "";
			boolean printed = false;
			for (String feature: features){
				Boolean inDict = true;
				Boolean isDigit = false;
				Boolean isLetter = false;
				for (char c: feature.toCharArray()){
					if (c >= '0' && c <= '9'){
						isDigit = true;
						break;
					}
					else if (c >= 'a' && c <= 'z'){
						isLetter = true;
						break;
					}
					else if (!set.contains(String.valueOf(c))){
						inDict = false;
						break;
					}
				}
				if (isDigit || isLetter || inDict){
					String newFeature = "";
					if (isDigit) newFeature = "0";
					else if (isLetter) newFeature = "a";
					else if (inDict) newFeature = feature;
					if (!newFeature.equals(lastFeature)){
						writer.print(newFeature + "|");
						printed = true;
					}
					lastFeature = newFeature;
				}
			}
			if (printed) writer.print("||"+label + "\n");
			line = br.readLine();
		}
		writer.close();
		br.close();
	}
}