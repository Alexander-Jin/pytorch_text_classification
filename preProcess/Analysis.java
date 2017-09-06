import java.util.*;
import java.io.*;

class Analysis{
	static class HospitalData{
		String hospital;
		int count;
		List<String> description;
		List<String> department;
		public HospitalData(String hospital){
			this.hospital = hospital;
			description = new ArrayList<String>();
			department = new ArrayList<String>();
			count = 0;
		}

		public int addData(String description, String department){
			this.description.add(description);
			this.department.add(department);
			count += 1;
			return count;
		}
	}

	public static void count(String fileName, int numTop) throws Exception {
		HashMap<String, HospitalData> map = new HashMap<String, HospitalData>();
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		br.readLine();
		String current = br.readLine();
		while (current != null){
			String[] row = current.split("\\|");
			String hospital = row[13];
			String description = row[8];
			String department = row[11];
			if (!department.contains("儿") && !department.contains("中医")){
				if (map.containsKey(hospital)){
					map.get(hospital).addData(description, department);
				}
				else{
					HospitalData newData = new HospitalData(hospital);
					newData.addData(description, department);
					map.put(newData.hospital, newData);
				}
			}

			current = br.readLine();
		}

		PriorityQueue<HospitalData> heap = new PriorityQueue<HospitalData>(
			new Comparator<HospitalData>(){
				public int compare(HospitalData h1, HospitalData h2){
					return h1.count - h2.count;
				}
			});

		for (HospitalData data: map.values()){
			heap.offer(data);
			if (heap.size() > numTop) heap.poll();
		}

		PrintWriter writer = new PrintWriter("count.txt", "UTF-8");
		while (!heap.isEmpty()){
			HospitalData data = heap.poll();
			PrintWriter hopspitalWriter = new PrintWriter(data.hospital + ".txt", "UTF-8");
			PrintWriter classWriter = new PrintWriter(data.hospital + "class.txt", "UTF-8");
			HashSet<String> set = new HashSet<String>();
			hopspitalWriter.println("description|department");
			for (int i = 0; i < data.count; i++){
				hopspitalWriter.print(data.description.get(i));
				hopspitalWriter.print("|");
				hopspitalWriter.println(data.department.get(i));
				if (!set.contains(data.department.get(i))) set.add(data.department.get(i));
			}
			hopspitalWriter.close();
			for (String s: set) classWriter.println(s);
			classWriter.close();
			writer.println(data.hospital + " | " + data.count);
		}
		writer.close();
	}

	public static void main(String[] args) throws Exception {
		if (args.length == 0) throw new Exception("Indicate file name");
		String fileName = args[0];
		int numTop = 50;
		if (args.length > 1) numTop = Integer.valueOf(args[1]);
		count(fileName, numTop);
	}
}