import java.util.*;
import java.io.*;

class Analysis{
	static class DataPoint{
		String department;
		String description;
		public DataPoint(String department, String description){
			this.department = department;
			this.description = description;
		}
	}

	static class DepartmentData{
		String department;
		List<DataPoint> data;
		int size;
		public DepartmentData(String department){
			this.department = department;
			data = new ArrayList<DataPoint>();
			size = 0;
		}

		public int addData(DataPoint dataPoint){
			this.data.add(dataPoint);
			size += 1;
			return size;
		}

		public int cleanData(){
			Collections.sort(data, new Comparator<DataPoint>(){
				public int compare(DataPoint d1, DataPoint d2){
					return d1.description.compareTo(d2.description);
				}
			});
			int dataSize = data.size();
			int i = 1;
			while (i < dataSize){
				if (i >= data.size()) break;
				if (data.get(i).equals(data.get(i - 1))) data.remove(i);
				else if (data.get(i).description.length() > 15 && data.get(i - 1).description.length() > 15){
					if (data.get(i).description.substring(0, 15).equals(data.get(i - 1).description.substring(0, 15))){
						if (data.get(i).description.length() < data.get(i - 1).description.length()){
							data.remove(i);
						}
						else data.remove(i - 1);
					}
					else i += 1;
				}
				else i += 1;
			}
			size = data.size();
			return size;
		}
	}

	public static void process(String fileName) throws Exception {
		HashMap<String, DepartmentData> map = new HashMap<String, DepartmentData>();
		HashMap<String, Integer> rawMap = new HashMap<String, Integer>();
		List<Integer> sentenceCount = new ArrayList<Integer>();
		HashMap<String, List<Integer>> sentenceCountMap = new HashMap<String, List<Integer>>();
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String current = br.readLine();

		while (current != null){
			if (current.length() < 10){
				current = br.readLine();
				continue;
			}
			//System.out.println(current);
			String[] row = current.split("\t");
			String description = row[0].trim();
            String department = row[row.length - 1].trim();
			description = description.replaceAll("([,?!\\|\\(\\)\\-/  。、？！（）；*~·]+[ ]*|(<[\\w=]+>)+)", "，");
			description = description.replaceAll("[\\d]+\\.[ ]?", "，");
			description = description.replaceAll("[，\\.]{2,}", "，");
			int titleIndex = description.indexOf(":");
			if (titleIndex > 0 && titleIndex < 12) description = description.substring(titleIndex + 1, description.length());
			if (description.length() > 0 && description.charAt(0) == '，') description = description.substring(1, description.length());
			if (description.length() > 0 && description.charAt(description.length() - 1) == '，') description = description.substring(0, description.length() - 1);
			if (description.length() > 10){
				if (map.containsKey(department)) map.get(department).addData(new DataPoint(department, description));
				else{
					DepartmentData d = new DepartmentData(department);
					d.addData(new DataPoint(department, description));
					map.put(department, d);
				}
			}
			rawMap.put(department, rawMap.getOrDefault(department, 0) + 1);
			current = br.readLine();
		}

		PrintWriter writer = new PrintWriter(new File("classDistribution.csv"));
		writer.println("department,raw,cleaned,removed");
		for (Map.Entry<String, DepartmentData> entry: map.entrySet()){
			String department = entry.getKey();
			writer.print(department);
			writer.print(",");
			writer.print(rawMap.get(department));
			writer.print(",");
			writer.print(entry.getValue().size);
			writer.print(",");
			entry.getValue().cleanData();
			writer.println(entry.getValue().size);
		}
		writer.close();

		writer = new PrintWriter(new File("dataset.txt"));
		writer.println("description|department");
		PrintWriter classWriter = new PrintWriter(new File("classes.txt"));
		for (Map.Entry<String, DepartmentData> entry: map.entrySet()){
			String department = entry.getKey();
			classWriter.println(department);
			for (DataPoint point: entry.getValue().data){
				writer.println(point.description + "|" + point.department);
			}
		}
		writer.close();
		classWriter.close();
	}

	public static void main(String[] args) throws Exception {
		String fileName = "transferedData.txt";
		if (args.length > 0) fileName = args[0];
		process(fileName);
	}
}