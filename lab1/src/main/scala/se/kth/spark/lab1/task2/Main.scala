package se.kth.spark.lab1.task2

import se.kth.spark.lab1._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.{Pipeline, PipelineModel}

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.linalg.Vector

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/main/resources/millionsong.txt"
    var df = sc.textFile(filePath).toDF()
    df.show()
    df.describe().show()
    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("value")
      .setOutputCol("tokenized")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    var tokenized = regexTokenizer.transform(df)
    tokenized.select("tokenized").take(5).foreach(println)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
    val arr2Vect = new Array2Vector()
                  .setInputCol(regexTokenizer.getOutputCol)
                  .setOutputCol("arr2Vect")
            //      .transform(tokenized)
    
    //arr2Vect.show()
    
    //Step4: extract the label(year) into a new column
    val lSlicer = new VectorSlicer().setInputCol("arr2Vect").setOutputCol("year").setIndices(Array(0))//.transform(arr2Vect)//.transform(arr2Vect)
//    
//    lSlicer.show()
//    lSlicer.printSchema()
   //   arr2Vect.select("arr2Vect").take(3).foreach(println)
    
//
    //Step5: convert type of the label from vector to double (use our Vector2Double)
    val udf = (x:Vector) => x(0)
    val v2d = new Vector2DoubleUDF(udf).setInputCol("year").setOutputCol("label")//.transform(lSlicer)
    
//    v2d.show()
//    v2d.printSchema()

//    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 
    val udf2 = (x:Double) => x  
    val lShifter = new DoubleUDF(udf2).setInputCol("label").setOutputCol("year_label")//.transform(v2d)
//    
//    lShifter.show()
//    lShifter.printSchema()

//    //Step7: extract just the 3 first features in a new vector column
    val fSlicer = new VectorSlicer().setInputCol("arr2Vect").setOutputCol("features").setIndices(Array(1,2,3))//.transform(lShifter)
//    fSlicer.show()
//    fSlicer.printSchema()
//    
//    val clean = fSlicer.drop("value").drop("tokenized").drop("arr2vect").drop("year").drop("label")
//    val newNames = Seq("label", "features")
//    val pipez = clean.toDF(newNames: _*)
//    
//    val labels = pipez.drop("features")
//    val features = pipez.drop("label")
//    
//    pipez.show()
//    pipez.printSchema()
//    pipez.take(5).foreach(println)
//
  //    //Step8: put everything together in a pipeline

   val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, fSlicer))
//
//    //Step9: generate model by fitting the rawDf into the pipeline
   val pipelineModel = pipeline.fit(df)
//   pipelineModel.show()
   println("such model::")
   pipelineModel.explainParams()
//
//    //Step10: transform data with the model - do predictions
    // ???? WHAT *Creates datafram* "Do predictions" - are you kidding me for real? With what?
   val test = createDataFrame(Seq(
  (2001.0, (0.555, 0.34234, 0.2342342)),
  (2010.0, (0.4353, 0.3234, 0.345345)),
  (2007.0, (0.2353, 0.23434, 0.75345))
  )).toDF("year_label", "features")

   pipelineModel.transform(test).select("year_label", "features").collect()
  .foreach { case Row(id: Double, text: Vector, prob: Vector, prediction: Double) =>
    println(s"($id, $text) --> prob=$prob, prediction=$prediction")
  }
   
//    //Step11: drop all columns from the dataframe other than label and features
 
  }
}