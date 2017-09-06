import java.util.*;
import java.io.*;

class ClassDistribution{

	public static void count(File f, PrintWriter cWriter) throws Exception {
		HashMap<String, Integer> map = new HashMap<String, Integer>();

		BufferedReader br = new BufferedReader(new FileReader(f));
		br.readLine();
		String current = br.readLine();
		while (current != null && current.length() > 0){
			String hospital = current.split("\\|")[1].trim();
			map.put(hospital, map.getOrDefault(hospital, 0) + 1);
			current = br.readLine();
		}
		br.close();

		PriorityQueue<Map.Entry<String, Integer>> heap = new PriorityQueue<Map.Entry<String, Integer>>(
			new Comparator<Map.Entry<String, Integer>>(){
				public int compare(Map.Entry<String, Integer> h1, Map.Entry<String, Integer> h2){
					return h1.getValue() - h2.getValue();
				}
			});
		for (Map.Entry<String, Integer> entry: map.entrySet()){
			heap.offer(entry);
		}

		String fileName = f.getName();
		PrintWriter writer = new PrintWriter(fileName.substring(0, fileName.length() - 4) + "class.txt", "UTF-8");
		int count = 0;
		while (!heap.isEmpty()){
			Map.Entry<String, Integer> entry = heap.poll();
			if (entry.getValue() > 50){
				writer.print(entry.getKey());
				writer.print('\n');
				count += entry.getValue();
			}
		}
		cWriter.print(fileName.substring(0, fileName.length() - 4) + ": ");
		cWriter.println(count);
		writer.close();
	}

	public static void main(String[] args) throws Exception {
		File folder = new File("./");
		File[] listOfFiles = folder.listFiles();
		PrintWriter writer = new PrintWriter("datasize.txt", "UTF-8");
		for (File f: listOfFiles){
			if (f.getName().endsWith(".txt")) count(f, writer);
		}
		writer.close();
	}
}