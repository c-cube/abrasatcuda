#!/usr/bin/env runhaskell

module Check
    (main,
    ) where

import System.Environment (getArgs)
import Control.Monad
import Data.List (isPrefixOf)
import Data.Maybe (isJust)


data Var a = Pos a | Neg a | Tauto
    deriving (Show,Eq)

instance Functor Var where
    f `fmap` Pos i = Pos (f i)
    f `fmap` Neg i = Neg (f i)
    _ `fmap` Tauto  = Tauto



main = do
    filename <- getName `liftM` getArgs 
    lines <- lines `liftM` readFile filename 

    let varNum = head $ (lines >>= getVarNum)

    let clauses = parse lines
    mapM_ (putStrLn . show) clauses
    print $ solve clauses varNum

  where
    getName :: [String] -> String
    getName []         = "tests/example.cnf"
    getName (f : _)    = f

    getVarNum line = if "p cnf" `isPrefixOf` line
        then return $ read . head $ drop 2 (words line)
        else []

-- assumes that there is a clause by line
parse :: [String] -> [[Var Int]]
parse lines = do 
    line <- (filter elimBadLines lines)
    return $ do
        token <- words line
        case (read token :: Int) of
            0         -> []
            n | n < 0 -> return $ Neg (abs n)
            n         -> return $ Pos n
  where
    elimBadLines :: String -> Bool
    elimBadLines line = not ("c" `isPrefixOf` line) && not ("p" `isPrefixOf` line)





-- check if a clause is satisfiable
check :: [[Var Int]] -> Maybe [[Var Int]]
check clauses = sequence (map checkClause clauses)
  where
    checkClause []     = Nothing
    checkClause clause | any (== Tauto) clause  = return [Tauto]
    checkClause clause                          = return clause


-- perform affectation of a truth value
subst :: [[Var Int]] -> Int -> Bool -> [[Var Int]]
subst clauses var truth = (flip map) clauses $ \ clause -> 
  do
    atom <- clause
    smallSubst atom
  where
    smallSubst (Pos v) | v == var   = if truth then [Tauto] else []
    smallSubst (Neg v) | v == var   = if truth then [] else [Tauto]
    smallSubst x                    = [x]


-- tries every possibility
solve :: [[Var Int]] -> Int -> Bool
solve clauses varNum = 
    isJust $ step clauses [1..varNum] 
  where
    step clauses []   = check clauses
    step clauses vars = do
        clauses <- check clauses -- check if not incoherent
        step (subst clauses curVar True) remainingVars `mplus`
            step (subst clauses curVar False) remainingVars
      where
        curVar        = head vars
        remainingVars = tail vars
        
    

