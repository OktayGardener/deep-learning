# Hello World program in Scala
object HelloWorld {
    def main(args: Array[String]) {
        println("Hello, World!")
    }
}

# Interactive
scala> object HelloWorld {
| def main(args: Array[String]) {
|   println("Hello, world!")
|  }
|}

defined module HelloWorld
scala> HelloWorld.main(null)
Hello, world!

scala>:q
>

# Compilation
scalac HelloWorld.scala
scalac -d classes HelloWorld.scala


# Script run w/ .sh script

# script.sh
#!/bin/bash
exec scala $0 $@
!#
object HelloWorld {
    def main(args: Array[String]) {
        println("Hello, world!")
    }
}

HelloWorld.main(null)
## Execute dat shiet!
> ./script.sh


# Scala Variables
Values: Immutable
Variables: Mutable

var myVar: Int = 0
val myVal: Int = 1

// Scala figures out the type of variables based on the assigned Values
var myVar = 0
val myVal = 1

// If the initial values are not assigned, explicit typing!
var myVar: Int
val myVal: Int


# If/Else
var x = 30;

if(x == 10) {
    println("Value of x is: 10")
} else if(x == 20) {

} else {

}

# Loops 1/3
var a = 10

do {
    println("Value of a: " + a)
    a = a + 1
} while(a < 20)

while(a < 20) {
    println("Value of a: " + a)
    a = a + 1
}


var a = 0
var b = 0

for(a <- 1 to 3; b <- 1 until 3) {
    println("Value of a: " + a + ", b" + b)
}

var numList = List(1,2,3,4,5,6)

for(a <- numList) {
    println("Value of a: " + a)
}

for(a <- numList if a != 3; if a < 5) {
    println("Value of a: " + a)
}

// for loop with yield
// store return values from a foor lop in a variable

var retVal = for(a <- numList if a != 3; if a < 6) yield a
println("")

# Handle exceptions

import java.io.FileReader
import java.io.FileNotFoundException
import java.io.IOException

object Test {
    def main(args: Array[String]) {
        try {
            val f = new FileReader("input.txt")
        } catch {
            case ex: FileNotFoundException => { println("Missing file exception") }
            case ex: IOException => {println("IO exception") }
        } finally { println("Exiting finally..") }
    }
}

# Scala Functions

def funcName([lsit of params]) : [return type] = {
    function body
    return [expr]
}

def addInt(a: Int, b: Int): Int = {
    var sum: Int = 0
    sum = a + b
    sum
}

println("Ret value: " + addInt(5,7))

### Default param values

def addInt(a: Int = 5, b: Int = 7): Int = {
    var sum: Int = 0
    sum = a + b
    return sum
}

println("Ret value: " + addInt() )
