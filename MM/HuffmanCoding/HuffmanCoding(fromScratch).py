"""
By Eden Zhou
Jan. 15, 2023
"""

class Node:
    """
    Huffman tree node definition.
    """
    def __init__(self, symbol=None, count=0, left=None, right=None):
        """
        initialization
          symbol   : symbol to be coded
          count    : count of symbol
          left     : left child node
          right    : right child node
        """
        self.__left = left
        self.__right = right
        self.__symbol = symbol
        self.__count = count
        self.__code_word = ''

    def setLeft(self, l):
        """
        sets the left child of current node
        """
        self.__left = l

    def setRight(self, r):
        """
        sets the right child of current node
        """
        self.__right = r

    def getLeft(self):
        """
        returns the left child of current node
        """
        return self.__left

    def getRight(self):
        """
        returns the right child of current node
        """
        return self.__right

    def setSymbol(self, symbol):
        """
        sets coding symbol of current node
        """
        self.__symbol = symbol

    def getSymbol(self):
        """
        returns coding symbol of current node
        """
        return self.__symbol

    def setCount(self, count):
        """
        sets count of the symbol
        """
        self.__count = count

    def getCount(self):
        """
        returns count of the symbol
        """
        return self.__count

    def setCodeWord(self, code_word):
        """
        sets code-word of the symbol
        """
        self.__code_word = code_word

    def getCodeWord(self):
        """
        returns code-word of the symbol
        """
        return self.__code_word

    def __lt__(self, node):
        return self.__count < node.getCount()

    def __repr__(self):
        return "symbol: {}, count: {}, code-word: {}".format(self.__symbol, self.__count, self.__code_word)


def addNode(word_dict):
    all_nodes = []  # put all nodes in all_nodes list
    nodes_count = []  # put all counts of the nodes in nodes_count list
    for i in word_dict.keys():
        node = Node(i, word_dict[i], None, None)  # build the node
        nodes_count.append(word_dict[i])  # append very count of the node to nodes_count list
        all_nodes.append(node)  # append every node to all_nodes list
    print("....")
    print(all_nodes)
    print(nodes_count)
    return all_nodes, nodes_count


def searchCodeWord(root, symbol):
    if root is None:  # if no root
        return None
    if root.getLeft() is None and root.getRight() is None:  # if leaf node
        if root.getSymbol() == symbol:
            return root.getCodeWord()
    if root.getLeft() != None:  # if have left leaf node
        code_word = searchCodeWord(root.getLeft(), symbol)  # recursion
        if code_word != None:
            return code_word
    if root.getRight() != None:  # if have right leaf node
        code_word = searchCodeWord(root.getRight(), symbol)  # recursion
        if code_word != None:
            return code_word


def searchSymbol(huffman_tree, code_word):
    if huffman_tree == None:  # if no root
        return None
    if huffman_tree.getLeft() is None and huffman_tree.getRight() is None:  # if leaf node
        if huffman_tree.getCodeWord() == code_word:
            return huffman_tree.getSymbol()
    if huffman_tree.getLeft() != None:  # if have left leaf node
        symbol = searchSymbol(huffman_tree.getLeft(), code_word)  # recursion
        if symbol != None:
            return symbol
    if huffman_tree.getRight() != None:  # if have right leaf node
        symbol = searchSymbol(huffman_tree.getRight(), code_word)  # recursion
        if symbol != None:
            return symbol

def treeCodesParser(root, token):
  try:
    assert root is not None
  except AssertionError:
    return None
  if root.getLeft() is not None:
    code_word = treeCodesParser(root.getLeft(), token)
    if code_word is not None:
      return code_word
  elif root.getRight() is not None:
    code_word = treeCodesParser(root.getRight(), token)
    if code_word is not None:
      return code_word
  else:
    if root.getSymbol() == token:
      return root.getCodeWord()

def buildDictionary(message):
    """
    counts the occurrence of every symbol in the message and store it in a python dictionary
      parameter:
        message: input message string
      return:
        python dictionary, key = symbol, value = occurrence
    """
    dictionary = {}
    for symbol in message:  # for every symbol in message
        if symbol not in dictionary:  # dictionary does not have the symbol
            dictionary[symbol] = 1
        else:  # dictionary has the symbol
            dictionary[symbol] = dictionary[symbol] + 1
    print(dictionary)
    return dictionary


def buildHuffmanTree(word_dict):
    """
    uses the word dictionary to generate a huffman tree using a min heap
      parameter:
        word_dict  : word dictionary generated by buildDictionary()
      return:
        root node of the huffman tree
    """
    all_nodes, nodes_count = addNode(word_dict)
    while len(all_nodes) > 1:
        nodes_count.sort()  # Sort the nodes_count list
        node_count1 = nodes_count[0]  # take out the smallest (first smallest) count of the node
        for node in all_nodes:
            if node.getCount() == node_count1:  # find the corresponding node
                min_node1 = node
                nodes_count.remove(node_count1)  # remove the first corresponding count of node in nodes_count
                all_nodes.remove(min_node1)  # remove the corresponding node in all_nodes
                break  # if we find a node which meet our requirement then stop
        node_count2 = nodes_count[0]  # take out the smallest (second smallest) count of the node
        for node in all_nodes:
            if node.getCount() == node_count2:  # find the corresponding node
                min_node2 = node
                nodes_count.remove(node_count2)  # remove the first corresponding count of node in nodes_count
                all_nodes.remove(min_node2)  # remove the corresponding node in all_nodes
                break  # if we find a node which meet our requirement then stop
        new_value = min_node1.getCount() + min_node2.getCount()  # get the new value
        internal_node = Node(None, new_value, min_node1, min_node2)  # build the internal node
        all_nodes.append(internal_node)  # append the internal node to the all_nodes list
        nodes_count.append(internal_node.getCount())  # append the count of the internal node to the nodes_count list
    print(all_nodes)
    print(nodes_count)
    return all_nodes[0]  # when there is only one node in the all_nodes list, return it


def assignCodeWord(root, code_word=''):
    """
    recursively assigns code-word to the nodes in the huffman tree
      parameter:
        root       : root node of the huffman tree
        code_word  : code-word for the root node
      return:
        no return
    """
    root.setCodeWord(code_word)  # set root's code word
    if root.getSymbol() is None:  # if the root is internal node
        left_code_word = code_word + '0'
        assignCodeWord(root.getLeft(), left_code_word)  # go to left node's huffman tree, recursion
        right_code_word = code_word + '1'
        assignCodeWord(root.getRight(), right_code_word)  # go to right node's huffman tree, recursion


def huffmanEncode(message):
    """
    converts the input message into huffman code
      parameter:
        message    : input message string
      return:
        a tuple, the first element is the huffman code string for the input message,
        the second element is the huffman tree root node
    """
    encode = ''  # initial encode
    dictionary = buildDictionary(message)  # build dictionary
    root = buildHuffmanTree(dictionary)  # build huffman tree
    assignCodeWord(root, '')  # assign codeword
    for symbol in message:
        string = searchCodeWord(root, symbol)  # search codeword
        encode = encode + string  # get the encode
    return encode, root


def huffmanDecode(message, huffman_tree):
    """
    decode the message
      parameter:
        message      : input huffman code string
        huffman_tree : huffman tree root node
      return:
        decoded message
    """
    decode = ''  # initial decode
    code_word = ''  # initial code_word
    if message == '':  # if huffman tree just has one node, we define the huffman code = ''
        count = huffman_tree.getCount()
        for i in range(count):
            decode = decode + huffman_tree.getSymbol()
    for number in message:
        code_word = code_word + number  # find a code_word that has corresponding symbol
        string = searchSymbol(huffman_tree, code_word)  # search the symbol
        if string != None:
            decode = decode + string  # get the decoded word
            code_word = ''  # reset the code word to ''
    return decode


def main():
    """
    main process goes here
    """
    message = input("Enter a message: ")
    encoded, rootNode = huffmanEncode(message)
    decoded = huffmanDecode(encoded, rootNode)
    print("Encode the message, and the huffman code is: ", encoded)
    print("Huffman code's length is: ", len(encoded))
    print("Decode the huffman code, and the decoded message is: ", decoded)



if __name__ == "__main__":
    main()
