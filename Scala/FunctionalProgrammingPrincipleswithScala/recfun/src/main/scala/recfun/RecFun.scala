package recfun

// import recfun.Main.{countChange, pascal}

object RecFun extends RecFunInterface {

  def main(args: Array[String]): Unit = {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(s"${pascal(col, row)} ")
      println()
    }
  }

  /**
   * Exercise 1
   */
  def pascal(c: Int, r: Int): Int = {

    if (c == r ){1}

    else if (c == 0){1}

    else {pascal(c-1,r-1)+pascal(c,r-1)}
  }

  /**
   * Exercise 2
   */
  def balance(chars: List[Char]): Boolean = {
    def sumUnpairedBrackets(char_list: List[Char],current_Sum: Int): Boolean = {


      if (char_list.isEmpty) {current_Sum == 0}

      else if (char_list.head == '(') {
        sumUnpairedBrackets(char_list.tail, current_Sum + 1)
      }

      else if (char_list.head == ')') {
        current_Sum> 0 && sumUnpairedBrackets(char_list.tail, current_Sum - 1)
      }

      else {sumUnpairedBrackets(char_list.tail,current_Sum)}
    }
    sumUnpairedBrackets(chars,0)
  }

  /**
   * Exercise 3
   */
  def countChange(money: Int, coins: List[Int]): Int = {

    if (money < 0 || coins.isEmpty ){0}

    else if (money  == 0){1}

    else {countChange(money-coins.head,coins) + countChange(money,coins.tail)}


  }

}
