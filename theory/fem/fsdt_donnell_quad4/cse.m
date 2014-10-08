(* ::Package:: *)

(*
Original: Stonewall Ballard
        http://stoney.sb.org/wordpress/2009/06/converting-symbolic-mathematica-expressions-to-c-code/
Modification: Eric Cousineau (eacousineau@gmail.com)
        . Reformatted code, placed it in a package
        . Fixed a small bug allowing it to return expressions with no subexpressions
        . Changed variable numbering to any number of digits
        . Fixed small bug in listSubExps[] handling if it returns no results from Scan[]
*)
BeginPackage["CommonSubexpressionElimination`"];
 
hoistCommonSubexpressions::usage = "Optimize expression by finding repeated subexpressions and extracting them into a set of rules.
Returns { {main_expr}, {subexpr -> var, ...} }";
 
Begin["`Private`"];
 
listSubExps[exp_]:= Module[{res},
Sort[
        res = Reap[
                Scan[
                        If[!AtomQ[#], Sow[#]]&,
                        exp, Infinity
                ]
        ];
        If[Length[res[[2]]] > 0, res[[2, 1]], {}],
        (*res[[2,1]],*)
        LeafCount[#1] < LeafCount[#2]&
]];
 
replaceRepeats[expr_]:=
Module[{ceqns, eqns, rules},
        ceqns = Reap[
                Module[{vnum=0, modExpr=expr, subExprs=listSubExps[expr], subExpr, mapping},
                        While[subExprs != {},
                                subExpr = First[subExprs];
                                subExprs = Rest[subExprs];
                                (* If the expression appears more than once *)
                                If[Count[subExprs, subExpr]>0,
                                        mapping = subExpr->Symbol["vv"<>IntegerString[vnum++,10]];
                                        Sow[mapping];
                                        modExpr=modExpr/.mapping;
                                        subExprs=DeleteCases[subExprs,subExpr]/.mapping
                                ]
                        ];
                modExpr
                ]
        ];
        eqns = ceqns[[1]];
        rules = ceqns[[2]];
        (* Change:
        Before, used Extract[temp, {{1}, {2, 1}}.
        If no expressions are found, it returns {}. If found, returns {{exprs...}}
        *)
        rules = If[Length[rules] > 0, rules[[1]], {}];
        Return[{eqns, rules}];
]
 
foldOneUseVariables[ceqns:{eqns_,rules_}]:=
Module[{newRules,updates},
        {newRules, updates} = Reap[
                Select[rules,
                        If[Count[ceqns, #[[2]], \[Infinity]]==2, Sow[#[[2]]->#[[1]]]; False, True]&
                ]
        ];
        {eqns,newRules}//.Flatten[updates]
]
 
renumberVariables[ceqns:{_,rules_}]:=
Module[{vnum=0},
        ceqns /. Map[#[[2]]->Symbol["v"<>IntegerString[vnum++]]&,rules]
]
 
hoistCommonSubexpressions[expr_]:=
renumberVariables[
        foldOneUseVariables[
                replaceRepeats[
                        Together[expr]
                ]
        ]
]
 
End[];
EndPackage[];
