package se.kth.spark.lab1.task6

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions._

import org.apache.spark.hack._
import org.apache.spark.sql.Row
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.linalg.Matrices
import org.apache.spark.mllib.evaluation.RegressionMetrics

case class Instance(label: Double, features: Vector)

object Helper {
  def rmse(labelsAndPreds: RDD[(Double, Double)]): Double = {
    // foreach (map) pow(label - prediction, 2) div n, reduce sum
    val rmse = labelsAndPreds.map(x => Math.pow(x._1 - x._2, 2)).reduce(_+_)
    return scala.math.sqrt(rmse/labelsAndPreds.count)
  }

  def predictOne(weights: Vector, features: Vector): Double = {
    var result: Double = 0.0
    
    for(i <- 0 to (weights.size - 1)) result += weights.apply(i) * features.apply(i)
    
    return result
  }

  def predict(weights: Vector, data: RDD[Instance]): RDD[(Double, Double)] = {
    return data.map(d => (d.label, predictOne(weights, d.features)))
  }
}

class MyLinearRegressionImpl(override val uid: String)
    extends MyLinearRegression[Vector, MyLinearRegressionImpl, MyLinearModelImpl] {

  def this() = this(Identifiable.randomUID("mylReg"))

  override def copy(extra: ParamMap): MyLinearRegressionImpl = defaultCopy(extra)

  def gradientSummand(weights: Vector, lp: Instance): Vector = {
    val predict = Helper.predictOne(weights, lp.features)
    val before = lp.label
    return VectorHelper.dot(lp.features, (predict - before))
  }

  def gradient(d: RDD[Instance], weights: Vector): Vector = {
    return d.map(instance => gradientSummand(weights, instance)).reduce(VectorHelper.sum)
  }

  def linregGradientDescent(trainData: RDD[Instance], numIters: Int): (Vector, Array[Double]) = {

    val n = trainData.count()
    println("Training Data Size is " + n)
    val d = trainData.take(1)(0).features.size
    var weights = VectorHelper.fill(d, 0)
    val alpha = 1.0
    val errorTrain = Array.fill[Double](numIters)(0.0)

    for (i <- 0 until numIters) {
      //compute this iterations set of predictions based on our current weights
      val labelsAndPredsTrain = Helper.predict(weights, trainData)
      //compute this iteration's RMSE
      errorTrain(i) = Helper.rmse(labelsAndPredsTrain)
      println("RMSE iteration num "+i+": " + errorTrain(i))

      //compute gradient
      val g = gradient(trainData, weights)
      //update the gradient step - the alpha
      val alpha_i = alpha / (n * scala.math.sqrt(i + 1))
      val wAux = VectorHelper.dot(g, (-1) * alpha_i)
      //update weights based on gradient
      weights = VectorHelper.sum(weights, wAux)
    }
    (weights, errorTrain)
  }

  def train(dataset: Dataset[_]): MyLinearModelImpl = {
    println("Training")

    val numIters = 100

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), col($(featuresCol))).rdd.map {
        case Row(label: Double, features: Vector) =>
          Instance(label, features)
      }

    val (weights, trainingError) = linregGradientDescent(instances, numIters)
    new MyLinearModelImpl(uid, weights, trainingError)
  }
}

class MyLinearModelImpl(override val uid: String, val weights: Vector, val trainingError: Array[Double])
    extends MyLinearModel[Vector, MyLinearModelImpl] {

  override def copy(extra: ParamMap): MyLinearModelImpl = defaultCopy(extra)

  def predict(features: Vector): Double = {
    val prediction = Helper.predictOne(weights, features)
    prediction
  }
}