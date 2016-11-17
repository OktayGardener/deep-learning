package se.kth.spark.lab1.task7

import org.apache.spark._
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{RegexTokenizer, VectorSlicer}
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}
import org.apache.spark.ml.linalg.Vector

object Main {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("lab1bonus")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._


    val filePath = "hdfs://10.0.104.163:8020/Projects/datasets/million_song/csv/all.txt"
    val testFilePath = "hdfs://10.0.104.163:8020/Projects/datasets/million_song/csv/a.txt"
    val df = sparkContext.textFile(filePath).toDF("raw")
    
    val regexTokenizer = new RegexTokenizer().setInputCol("raw").setOutputCol("tokenized").setPattern(",")
    val arr2Vect = new Array2Vector().setInputCol("tokenized").setOutputCol("allFeatures")
    val lSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("yearv").setIndices(Array(0))
    
    val udf = (x:Vector) => x(0)
    val v2d = new Vector2DoubleUDF(udf).setInputCol("yearv").setOutputCol("year2d")
    
    val minYear = 1922.0
    val udf2: Double => Double = {_ - minYear }
    
    val lShifter = new DoubleUDF(udf2).setInputCol("year2d").setOutputCol("label")
    val fSlicer = new VectorSlicer().setInputCol("allFeatures").setOutputCol("features").setIndices(Array(1,2,3))

    val myLR = new LinearRegression().setMaxIter(10).setRegParam(0.1).setElasticNetParam(0.1);
    
    var lrStages = Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR)
    val pipeline = new Pipeline().setStages(lrStages)
    val pipelineModel: PipelineModel = pipeline.fit(df)
    val lrModel = pipelineModel.stages(6).asInstanceOf[LinearRegressionModel]
    println("rootMeanSquaredError: " + lrModel.summary.rootMeanSquaredError)

    val testDF = sparkContext.textFile(testFilePath).toDF("raw")
    var res = pipelineModel.transform(testDF)
    res.select("label", "features", "prediction").show(50)
    sc.stop()
  }
}