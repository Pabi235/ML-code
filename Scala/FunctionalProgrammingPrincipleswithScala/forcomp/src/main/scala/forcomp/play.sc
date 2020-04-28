val x = List(('a',1),('a',2),('a',3),('a',4)
  ,('a',5),('a',6),('a',7),('a',8),('a',9))
val y =  List(('a',1),('a',2),('a',3),('a',4))

//x diff y

val lard = List(('a', 1), ('d', 2), ('l', 1), ('r', 1))
val r = List(('r', 1))
//val lad = List(('a', 1), ('d', 1), ('l', 1))

val alpha = List('a','b','c')

//val chr_to_Index = alpha.zipWithIndex.map{ case (v,i) => (v,i) }.toMap

//alpha(chr_to_Index('a'))
//alpha(chr_to_Index.get('a'))
//for((e,i) <- List("Mary", "had", "a", "little", "lamb").zipWithIndex)

val y_chrs = r.map(_._1)
val y_intgrs = r.map(_._2)

val chr_to_Index = y_chrs.zipWithIndex.map{ case (v,i) => (v,i) }.toMap.withDefaultValue(0)

for(pair_x <-lard
     if !(y_chrs.contains(pair_x._1)  & pair_x._2 <= y_intgrs(chr_to_Index(pair_x._1)))
) yield pair_x

//lard.map(_._1)
//val intgrs4 = lard.map(_._2)

/*
type Occurrences = List[(Char, Int)]

//val intgr3 = 3

//'a' < 'b'

val k = List(('a',2),('b',2))

val singles = k.foldLeft(List[(Char,Int)]())(
  (Acc,next) =>Acc:::(for (i <-1 to next._2) yield (next._1, i)).toList
)


//for (x<-singles;
//     y <- singles
//     if x._1 < y._1
//) yield List(x,y)

//singles.map(List(_))


k.foldRight(List[Occurrences](Nil)) { case (head_chr_int_pair, acc) => {
val chr_int_pairs = List(head_chr_int_pair).foldLeft(List[(Char,Int)]())(
  (Acc,next) =>Acc:::(for (i <-1 to next._2) yield (next._1, i)).toList)

  acc ++ ( for { comb <- acc; pair <- chr_int_pairs } yield pair :: comb )
} }


//List[List[(Char,Int)]]()
//(for (i <-1 to 3) yield ('a', i)).toList.map(List(_))

//List[List[(Char,Int)]]():::(for (i <-1 to 3) yield ('a', i)).toList.map(List(_))


//for(x <- singles;y <- List(('b',2))) yield List(x):::List(y)

//l foldRight(List[(Char,Int)]()){

//}

//singles.foldRight(List[List[(Char,Int)]]()) { case ((ch,tm), acc) => {
//  acc ++ ( for { comb <- acc; n <- 1 to tm } yield (ch, n) :: comb )
//} }


//singles.foldLeft(List[List[(Char,Int)]]())(
// (Acc,next) => //List(next)::Acc
// for( combination <- Acc;
//        pair <- List(List(next)) )yield List(combination,pair)

//)
*/
