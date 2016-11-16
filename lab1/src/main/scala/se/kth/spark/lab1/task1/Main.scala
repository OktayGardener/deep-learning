package se.kth.spark.lab1.task1

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    val rawDF = sc.textFile(filePath).toDF()

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    println("Features in text form:")
    rdd.take(5).foreach(println) // print 5 rows, delim: "," (csv), numfeatures: 13
    println("------------------------------")
    
    //Step2: split each row into an array of features
    println("Features to WrappedArray:")
    val recordsRdd = rdd.map(line => line.split(","))
 //   recordsRdd.foreach(println)
    println("------------------------------")
    //Step3: map each row into a Song object by using the year label and the first three features  
    val songsRdd = recordsRdd.map(
        attr => Obs(attr(0).toDouble, 
            attr(1).toDouble, 
            attr(2).toDouble, 
            attr(3).toDouble)
         )

    val df = songsRdd.toDF()
    
    println("------------------------------")
    println("DataFrame: ")
    df.take(5).foreach(println)
    
    println("DataFrame Schema: ")
    df.printSchema()
    
    println("DataFrame Describe: ")
    df.describe().show()
    
    val numSongs = df.count
    println("NumSongs: " + numSongs)
    
    val released98To00 = df.filter("year >= 1998 AND year <= 2000")
    println("Number of songs released from 1998 to 2000: " + released98To00.count)
    
    println("------------------------------")
    
    println("Mean for year row: ")
    df.select(mean("year")).show()
    
    println("Min for year row: ")
    df.select(min("year")).show()
      
    println("Max for year row: ")
    df.select(max("year")).show()
    
    //println(df.select("year >= 2000 AND year <= 2010").count)
    println("between year 2000 and 2010:")
    df.filter("year >= 2000 AND year <= 2010").show()
    
    println("random sample 2000 - 2010")
    df.filter("year >= 2000 AND year <= 2010").sample(true, df.count).show()

    
  }
}