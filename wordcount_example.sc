
// imports
import org.apache.spark._
import org.apache.spark.SparkContext._


object WordCount {

    def main(args: Array[String]) {
    
      // set file input and output locations

      val conf = new SparkConf()
      val sc = new SparkContext(conf)
      val input =  sc.textFile(args(0))
      
      // map and reduce by key
      val words = input.flatMap(line => line.split(" "))
      val counts = words.map(word => (word, 1)).reduceByKey{case (x, y) => x + y}
      
      // save to output
      counts.saveAsTextFile(args(1))
      
    }
}
