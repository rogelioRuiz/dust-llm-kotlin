package io.t6x.dust.llm

import java.util.Locale

data class ChatMessage(val role: String, val content: String)

class ChatTemplateEngine(templateString: String?) {
    private val template: String = templateString ?: CHAT_ML_TEMPLATE

    fun apply(
        messages: List<ChatMessage>,
        addGenerationPrompt: Boolean,
        bosToken: String = "",
        eosToken: String = "",
    ): String {
        val nodes = TemplateParser(template).parse()
        val rootContext = mutableMapOf<String, Any?>(
            "messages" to messages.map { mutableMapOf<String, Any?>("role" to it.role, "content" to it.content) },
            "add_generation_prompt" to addGenerationPrompt,
            "bos_token" to bosToken,
            "eos_token" to eosToken,
        )
        return TemplateEvaluator(rootContext).render(nodes)
    }

    companion object {
        const val CHAT_ML_TEMPLATE =
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    }
}

private class ChatTemplateException(message: String) : IllegalArgumentException(message)

private object Undefined

private class Namespace(
    val values: MutableMap<String, Any?> = mutableMapOf(),
)

private data class IfClause(
    val condition: Expression?,
    val body: List<Node>,
)

private sealed interface CallArgument {
    data class Positional(val expression: Expression) : CallArgument
    data class Named(val name: String, val expression: Expression) : CallArgument
}

private sealed interface LiteralValue {
    data class StringValue(val value: String) : LiteralValue
    data class IntValue(val value: Int) : LiteralValue
    data class BoolValue(val value: Boolean) : LiteralValue
}

private enum class UnaryOperator {
    NOT,
    NEGATE,
}

private enum class BinaryOperator {
    ADD,
    EQUALS,
    NOT_EQUALS,
    AND,
    OR,
    CONTAINS,
}

private sealed interface Expression {
    data class Literal(val value: LiteralValue) : Expression
    data class Variable(val name: String) : Expression
    data class Unary(val operator: UnaryOperator, val value: Expression) : Expression
    data class Binary(val operator: BinaryOperator, val left: Expression, val right: Expression) : Expression
    data class IsDefined(val value: Expression, val negated: Boolean) : Expression
    data class Member(val base: Expression, val name: String) : Expression
    data class Subscript(val base: Expression, val index: Expression) : Expression
    data class Slice(val base: Expression, val start: Expression?) : Expression
    data class Call(val callee: Expression, val arguments: List<CallArgument>) : Expression
    data class Filter(val base: Expression, val name: String) : Expression
}

private sealed interface Node {
    data class Text(val value: String) : Node
    data class Output(val expression: Expression) : Node
    data class ForLoop(
        val variable: String,
        val iterable: Expression,
        val body: List<Node>,
        val elseBody: List<Node>?,
    ) : Node
    data class Conditional(val clauses: List<IfClause>) : Node
    data class SetVariable(val name: String, val value: Expression) : Node
    data class SetAttribute(val target: String, val attribute: String, val value: Expression) : Node
}

private class TemplateParser(
    private val template: String,
) {
    private var index: Int = 0
    private var trimNextLeadingWhitespace: Boolean = false

    fun parse(): List<Node> = parseNodes(emptySet()).first

    private fun parseNodes(terminators: Set<String>): Pair<List<Node>, String?> {
        val nodes = mutableListOf<Node>()

        while (index < template.length) {
            when {
                matches("{{", index) -> {
                    val tag = readTag("{{", "}}")
                    if (tag.trimLeft) {
                        trimTrailingWhitespace(nodes)
                    }
                    nodes += Node.Output(ExpressionParser(tag.content).parse())
                    trimNextLeadingWhitespace = tag.trimRight
                }
                matches("{%", index) -> {
                    val tag = readTag("{%", "%}")
                    if (tag.trimLeft) {
                        trimTrailingWhitespace(nodes)
                    }
                    trimNextLeadingWhitespace = tag.trimRight

                    val statement = tag.content.trim()
                    val head = statementHead(statement)

                    if (head in terminators) {
                        return nodes to statement
                    }

                    nodes += when (head) {
                        "for" -> parseForStatement(statement)
                        "if" -> parseIfStatement(statement)
                        "set" -> parseSetStatement(statement)
                        else -> throw ChatTemplateException("Unsupported statement: $statement")
                    }
                }
                else -> {
                    var text = readText()
                    if (trimNextLeadingWhitespace) {
                        text = text.dropWhile { it.isWhitespace() }
                        trimNextLeadingWhitespace = false
                    }
                    if (text.isNotEmpty()) {
                        nodes += Node.Text(text)
                    }
                }
            }
        }

        if (terminators.isNotEmpty()) {
            throw ChatTemplateException("Missing terminator: ${terminators.sorted().joinToString(", ")}")
        }

        return nodes to null
    }

    private fun parseForStatement(statement: String): Node {
        val remainder = statement.removePrefix("for").trim()
        val (variable, iterableSource) = splitTopLevel(remainder, " in ")
            ?: throw ChatTemplateException("Invalid for statement: $statement")

        val variableName = variable.trim()
        if (!isValidIdentifier(variableName)) {
            throw ChatTemplateException("Invalid loop variable: $variableName")
        }

        val iterable = ExpressionParser(iterableSource).parse()
        val (body, terminator) = parseNodes(setOf("else", "endfor"))

        val elseBody = if (terminator != null && statementHead(terminator) == "else") {
            val (parsedElseBody, endTerminator) = parseNodes(setOf("endfor"))
            if (endTerminator == null || statementHead(endTerminator) != "endfor") {
                throw ChatTemplateException("Missing endfor")
            }
            parsedElseBody
        } else {
            if (terminator == null) {
                throw ChatTemplateException("Missing endfor")
            }
            null
        }

        return Node.ForLoop(variableName, iterable, body, elseBody)
    }

    private fun parseIfStatement(statement: String): Node {
        val clauses = mutableListOf<IfClause>()
        var condition = ExpressionParser(statement.removePrefix("if").trim()).parse()

        while (true) {
            val (body, terminator) = parseNodes(setOf("elif", "else", "endif"))
            clauses += IfClause(condition, body)

            val currentTerminator = terminator ?: throw ChatTemplateException("Missing endif")
            when (statementHead(currentTerminator)) {
                "elif" -> {
                    condition = ExpressionParser(currentTerminator.removePrefix("elif").trim()).parse()
                }
                "else" -> {
                    val (elseBody, endTerminator) = parseNodes(setOf("endif"))
                    if (endTerminator == null || statementHead(endTerminator) != "endif") {
                        throw ChatTemplateException("Missing endif")
                    }
                    clauses += IfClause(null, elseBody)
                    break
                }
                else -> break
            }
        }

        return Node.Conditional(clauses)
    }

    private fun parseSetStatement(statement: String): Node {
        val remainder = statement.removePrefix("set").trim()
        val (targetSource, valueSource) = splitTopLevel(remainder, " = ")
            ?: throw ChatTemplateException("Invalid set statement: $statement")

        val target = targetSource.trim()
        val value = ExpressionParser(valueSource).parse()
        val parts = target.split('.', limit = 2)

        return if (parts.size == 2) {
            if (!isValidIdentifier(parts[0]) || !isValidIdentifier(parts[1])) {
                throw ChatTemplateException("Invalid namespace assignment: $target")
            }
            Node.SetAttribute(parts[0], parts[1], value)
        } else {
            if (!isValidIdentifier(target)) {
                throw ChatTemplateException("Invalid variable assignment: $target")
            }
            Node.SetVariable(target, value)
        }
    }

    private fun readText(): String {
        val start = index
        while (index < template.length && !matches("{{", index) && !matches("{%", index)) {
            index += 1
        }
        return template.substring(start, index)
    }

    private fun readTag(open: String, close: String): Tag {
        index += open.length
        val trimLeft = index < template.length && template[index] == '-'
        if (trimLeft) {
            index += 1
        }

        val contentStart = index
        var quote: Char? = null
        var escaped = false

        while (index < template.length) {
            val character = template[index]

            if (quote != null) {
                if (escaped) {
                    escaped = false
                } else if (character == '\\') {
                    escaped = true
                } else if (character == quote) {
                    quote = null
                }
                index += 1
                continue
            }

            if (character == '\'' || character == '"') {
                quote = character
                index += 1
                continue
            }

            if (character == '-' && matches(close, index + 1)) {
                val content = template.substring(contentStart, index)
                index += 1 + close.length
                return Tag(content, trimLeft, true)
            }

            if (matches(close, index)) {
                val content = template.substring(contentStart, index)
                index += close.length
                return Tag(content, trimLeft, false)
            }

            index += 1
        }

        throw ChatTemplateException("Unterminated tag")
    }

    private fun trimTrailingWhitespace(nodes: MutableList<Node>) {
        val last = nodes.lastOrNull() as? Node.Text ?: return
        val trimmed = last.value.dropLastWhile { it.isWhitespace() }
        nodes.removeAt(nodes.lastIndex)
        if (trimmed.isNotEmpty()) {
            nodes += Node.Text(trimmed)
        }
    }

    private fun matches(value: String, position: Int): Boolean {
        if (position < 0 || position + value.length > template.length) {
            return false
        }
        return template.regionMatches(position, value, 0, value.length)
    }

    private fun statementHead(statement: String): String = statement.split(Regex("\\s+"), limit = 2).firstOrNull().orEmpty()

    private fun splitTopLevel(value: String, separator: String): Pair<String, String>? {
        if (value.length < separator.length) {
            return null
        }

        var quote: Char? = null
        var escaped = false
        var parenDepth = 0
        var bracketDepth = 0
        var position = 0

        while (position <= value.length - separator.length) {
            val character = value[position]

            if (quote != null) {
                if (escaped) {
                    escaped = false
                } else if (character == '\\') {
                    escaped = true
                } else if (character == quote) {
                    quote = null
                }
                position += 1
                continue
            }

            if (character == '\'' || character == '"') {
                quote = character
                position += 1
                continue
            }

            when (character) {
                '(' -> parenDepth += 1
                ')' -> parenDepth = (parenDepth - 1).coerceAtLeast(0)
                '[' -> bracketDepth += 1
                ']' -> bracketDepth = (bracketDepth - 1).coerceAtLeast(0)
            }

            if (parenDepth == 0 && bracketDepth == 0 && value.regionMatches(position, separator, 0, separator.length)) {
                return value.substring(0, position) to value.substring(position + separator.length)
            }

            position += 1
        }

        return null
    }

    private fun isValidIdentifier(value: String): Boolean {
        if (value.isEmpty()) {
            return false
        }
        val first = value.first()
        if (first != '_' && !first.isLetter()) {
            return false
        }
        return value.drop(1).all { it == '_' || it.isLetterOrDigit() }
    }

    private data class Tag(
        val content: String,
        val trimLeft: Boolean,
        val trimRight: Boolean,
    )
}

private sealed interface Token {
    data class Identifier(val value: String) : Token
    data class StringValue(val value: String) : Token
    data class IntegerValue(val value: Int) : Token
    data class Symbol(val value: String) : Token
    object End : Token
}

private class Tokenizer(
    private val source: String,
) {
    private var index: Int = 0

    fun tokenize(): List<Token> {
        val tokens = mutableListOf<Token>()

        while (true) {
            skipWhitespace()
            if (index >= source.length) {
                tokens += Token.End
                return tokens
            }

            val character = source[index]
            when {
                character == '\'' || character == '"' -> tokens += Token.StringValue(readString(character))
                character.isDigit() -> tokens += Token.IntegerValue(readInteger())
                character == '_' || character.isLetter() -> tokens += Token.Identifier(readIdentifier())
                match("==") -> tokens += Token.Symbol("==")
                match("!=") -> tokens += Token.Symbol("!=")
                character in setOf('(', ')', '[', ']', ':', ',', '.', '+', '|', '-', '=') -> {
                    index += 1
                    tokens += Token.Symbol(character.toString())
                }
                else -> throw ChatTemplateException("Unexpected token: $character")
            }
        }
    }

    private fun skipWhitespace() {
        while (index < source.length && source[index].isWhitespace()) {
            index += 1
        }
    }

    private fun readString(quote: Char): String {
        index += 1
        val builder = StringBuilder()

        while (index < source.length) {
            val character = source[index++]
            when {
                character == quote -> return builder.toString()
                character == '\\' -> {
                    if (index >= source.length) {
                        throw ChatTemplateException("Unterminated escape sequence")
                    }
                    when (val escaped = source[index++]) {
                        'n' -> builder.append('\n')
                        'r' -> builder.append('\r')
                        't' -> builder.append('\t')
                        '\\' -> builder.append('\\')
                        '\'', '"' -> builder.append(escaped)
                        else -> builder.append(escaped)
                    }
                }
                else -> builder.append(character)
            }
        }

        throw ChatTemplateException("Unterminated string literal")
    }

    private fun readInteger(): Int {
        val start = index
        while (index < source.length && source[index].isDigit()) {
            index += 1
        }
        return source.substring(start, index).toInt()
    }

    private fun readIdentifier(): String {
        val start = index
        while (index < source.length && (source[index] == '_' || source[index].isLetterOrDigit())) {
            index += 1
        }
        return source.substring(start, index)
    }

    private fun match(value: String): Boolean {
        if (index + value.length > source.length) {
            return false
        }
        if (!source.regionMatches(index, value, 0, value.length)) {
            return false
        }
        index += value.length
        return true
    }
}

private class ExpressionParser(
    source: String,
) {
    private val tokens: List<Token> = Tokenizer(source).tokenize()
    private var index: Int = 0

    fun parse(): Expression {
        val expression = parseOr()
        if (currentToken() != Token.End) {
            throw ChatTemplateException("Unexpected token in expression")
        }
        return expression
    }

    private fun parseOr(): Expression {
        var expression = parseAnd()
        while (matchIdentifier("or")) {
            expression = Expression.Binary(BinaryOperator.OR, expression, parseAnd())
        }
        return expression
    }

    private fun parseAnd(): Expression {
        var expression = parseNot()
        while (matchIdentifier("and")) {
            expression = Expression.Binary(BinaryOperator.AND, expression, parseNot())
        }
        return expression
    }

    private fun parseNot(): Expression {
        return if (matchIdentifier("not")) {
            Expression.Unary(UnaryOperator.NOT, parseNot())
        } else {
            parseComparison()
        }
    }

    private fun parseComparison(): Expression {
        var expression = parseAdditive()

        while (true) {
            expression = when {
                matchSymbol("==") -> Expression.Binary(BinaryOperator.EQUALS, expression, parseAdditive())
                matchSymbol("!=") -> Expression.Binary(BinaryOperator.NOT_EQUALS, expression, parseAdditive())
                matchIdentifier("in") -> Expression.Binary(BinaryOperator.CONTAINS, expression, parseAdditive())
                matchIdentifier("is") -> {
                    val negated = matchIdentifier("not")
                    if (!matchIdentifier("defined")) {
                        throw ChatTemplateException("Expected 'defined' after 'is'")
                    }
                    Expression.IsDefined(expression, negated)
                }
                else -> return expression
            }
        }
    }

    private fun parseAdditive(): Expression {
        var expression = parsePostfix()
        while (matchSymbol("+")) {
            expression = Expression.Binary(BinaryOperator.ADD, expression, parsePostfix())
        }
        return expression
    }

    private fun parsePostfix(): Expression {
        var expression = parsePrimary()

        while (true) {
            expression = when {
                matchSymbol(".") -> Expression.Member(expression, readIdentifier())
                matchSymbol("[") -> {
                    if (matchSymbol(":")) {
                        expectSymbol("]")
                        Expression.Slice(expression, null)
                    } else {
                        val start = parseOr()
                        if (matchSymbol(":")) {
                            expectSymbol("]")
                            Expression.Slice(expression, start)
                        } else {
                            expectSymbol("]")
                            Expression.Subscript(expression, start)
                        }
                    }
                }
                matchSymbol("(") -> Expression.Call(expression, parseCallArguments())
                matchSymbol("|") -> Expression.Filter(expression, readIdentifier())
                else -> return expression
            }
        }
    }

    private fun parsePrimary(): Expression {
        return when (val token = currentToken()) {
            is Token.StringValue -> {
                index += 1
                Expression.Literal(LiteralValue.StringValue(token.value))
            }
            is Token.IntegerValue -> {
                index += 1
                Expression.Literal(LiteralValue.IntValue(token.value))
            }
            is Token.Identifier -> {
                index += 1
                when (token.value) {
                    "true" -> Expression.Literal(LiteralValue.BoolValue(true))
                    "false" -> Expression.Literal(LiteralValue.BoolValue(false))
                    else -> Expression.Variable(token.value)
                }
            }
            is Token.Symbol -> when (token.value) {
                "(" -> {
                    index += 1
                    val expression = parseOr()
                    expectSymbol(")")
                    expression
                }
                "-" -> {
                    index += 1
                    Expression.Unary(UnaryOperator.NEGATE, parsePrimary())
                }
                else -> throw ChatTemplateException("Unexpected token in expression")
            }
            Token.End -> throw ChatTemplateException("Unexpected token in expression")
        }
    }

    private fun parseCallArguments(): List<CallArgument> {
        val arguments = mutableListOf<CallArgument>()
        if (matchSymbol(")")) {
            return arguments
        }

        while (true) {
            arguments += if (currentToken() is Token.Identifier && nextToken() == Token.Symbol("=")) {
                val name = (currentToken() as Token.Identifier).value
                index += 1
                expectSymbol("=")
                CallArgument.Named(name, parseOr())
            } else {
                CallArgument.Positional(parseOr())
            }

            if (matchSymbol(")")) {
                return arguments
            }
            expectSymbol(",")
        }
    }

    private fun currentToken(): Token = tokens[index]

    private fun nextToken(): Token = tokens[(index + 1).coerceAtMost(tokens.lastIndex)]

    private fun matchIdentifier(value: String): Boolean {
        val token = currentToken() as? Token.Identifier ?: return false
        if (token.value != value) {
            return false
        }
        index += 1
        return true
    }

    private fun matchSymbol(value: String): Boolean {
        val token = currentToken() as? Token.Symbol ?: return false
        if (token.value != value) {
            return false
        }
        index += 1
        return true
    }

    private fun expectSymbol(value: String) {
        if (!matchSymbol(value)) {
            throw ChatTemplateException("Expected '$value'")
        }
    }

    private fun readIdentifier(): String {
        val token = currentToken() as? Token.Identifier ?: throw ChatTemplateException("Expected identifier")
        index += 1
        return token.value
    }
}

private class TemplateEvaluator(
    rootContext: MutableMap<String, Any?>,
) {
    private val scopes = mutableListOf(rootContext)

    fun render(nodes: List<Node>): String {
        val builder = StringBuilder()
        render(nodes, builder)
        return builder.toString()
    }

    private fun render(nodes: List<Node>, builder: StringBuilder) {
        for (node in nodes) {
            when (node) {
                is Node.Text -> builder.append(node.value)
                is Node.Output -> builder.append(stringValue(evaluate(node.expression)))
                is Node.ForLoop -> {
                    val sequence = sequenceValues(evaluate(node.iterable))
                    if (sequence.isEmpty()) {
                        node.elseBody?.let { render(it, builder) }
                        continue
                    }

                    sequence.forEachIndexed { index, item ->
                        scopes += mutableMapOf(
                            node.variable to item,
                            "loop" to mutableMapOf<String, Any?>(
                                "index0" to index,
                                "first" to (index == 0),
                                "last" to (index == sequence.lastIndex),
                                "length" to sequence.size,
                            ),
                        )
                        render(node.body, builder)
                        scopes.removeAt(scopes.lastIndex)
                    }
                }
                is Node.Conditional -> {
                    for (clause in node.clauses) {
                        if (clause.condition == null || truthy(evaluate(clause.condition))) {
                            render(clause.body, builder)
                            break
                        }
                    }
                }
                is Node.SetVariable -> scopes.last()[node.name] = evaluate(node.value)
                is Node.SetAttribute -> {
                    val namespace = lookup(node.target) as? Namespace
                        ?: throw ChatTemplateException("Variable '${node.target}' is not a namespace")
                    namespace.values[node.attribute] = evaluate(node.value)
                }
            }
        }
    }

    private fun evaluate(expression: Expression): Any? {
        return when (expression) {
            is Expression.Literal -> when (val value = expression.value) {
                is LiteralValue.StringValue -> value.value
                is LiteralValue.IntValue -> value.value
                is LiteralValue.BoolValue -> value.value
            }
            is Expression.Variable -> lookup(expression.name)
            is Expression.Unary -> {
                val value = evaluate(expression.value)
                when (expression.operator) {
                    UnaryOperator.NOT -> !truthy(value)
                    UnaryOperator.NEGATE -> intValue(value)?.let { -it }
                        ?: throw ChatTemplateException("Unary minus expects an integer")
                }
            }
            is Expression.Binary -> {
                val left = evaluate(expression.left)
                when (expression.operator) {
                    BinaryOperator.AND -> if (!truthy(left)) false else truthy(evaluate(expression.right))
                    BinaryOperator.OR -> if (truthy(left)) true else truthy(evaluate(expression.right))
                    BinaryOperator.ADD -> {
                        val right = evaluate(expression.right)
                        if (left is Int && right is Int) {
                            left + right
                        } else {
                            stringValue(left) + stringValue(right)
                        }
                    }
                    BinaryOperator.EQUALS -> valuesEqual(left, evaluate(expression.right))
                    BinaryOperator.NOT_EQUALS -> !valuesEqual(left, evaluate(expression.right))
                    BinaryOperator.CONTAINS -> containsValue(left, evaluate(expression.right))
                }
            }
            is Expression.IsDefined -> {
                val isDefined = evaluate(expression.value) !== Undefined
                if (expression.negated) !isDefined else isDefined
            }
            is Expression.Member -> memberValue(evaluate(expression.base), expression.name)
            is Expression.Subscript -> subscriptValue(evaluate(expression.base), evaluate(expression.index))
            is Expression.Slice -> sliceValue(evaluate(expression.base), expression.start?.let { evaluate(it) })
            is Expression.Call -> call(expression.callee, expression.arguments)
            is Expression.Filter -> applyFilter(expression.name, evaluate(expression.base))
        }
    }

    private fun lookup(name: String): Any? {
        for (scope in scopes.asReversed()) {
            if (scope.containsKey(name)) {
                return scope[name]
            }
        }
        return Undefined
    }

    private fun truthy(value: Any?): Boolean {
        return when (value) {
            Undefined, null -> false
            is Boolean -> value
            is Int -> value != 0
            is String -> value.isNotEmpty()
            is List<*> -> value.isNotEmpty()
            is Map<*, *> -> value.isNotEmpty()
            is Namespace -> value.values.isNotEmpty()
            else -> true
        }
    }

    private fun stringValue(value: Any?): String {
        return when (value) {
            Undefined, null -> ""
            is String -> value
            is Int -> value.toString()
            is Boolean -> value.toString()
            else -> ""
        }
    }

    private fun intValue(value: Any?): Int? {
        return when (value) {
            is Int -> value
            is String -> value.toIntOrNull()
            else -> null
        }
    }

    private fun valuesEqual(left: Any?, right: Any?): Boolean {
        if (left === Undefined || right === Undefined) {
            return left === Undefined && right === Undefined
        }
        return left == right || stringValue(left) == stringValue(right)
    }

    private fun containsValue(value: Any?, container: Any?): Boolean {
        return when (container) {
            is String -> container.contains(stringValue(value))
            is List<*> -> container.any { valuesEqual(it, value) }
            is Map<*, *> -> container.containsKey(stringValue(value))
            is Namespace -> container.values.containsKey(stringValue(value))
            else -> false
        }
    }

    private fun memberValue(base: Any?, name: String): Any? {
        return when (base) {
            Undefined, null -> Undefined
            is Map<*, *> -> base[name] ?: Undefined
            is Namespace -> base.values[name] ?: Undefined
            else -> throw ChatTemplateException("Unsupported attribute access: $name")
        }
    }

    private fun subscriptValue(base: Any?, index: Any?): Any? {
        return when (base) {
            Undefined, null -> Undefined
            is Map<*, *> -> base[stringValue(index)] ?: Undefined
            is Namespace -> base.values[stringValue(index)] ?: Undefined
            is List<*> -> {
                val rawIndex = intValue(index) ?: throw ChatTemplateException("List index must be an integer")
                val resolvedIndex = if (rawIndex >= 0) rawIndex else base.size + rawIndex
                if (resolvedIndex !in base.indices) Undefined else base[resolvedIndex]
            }
            else -> throw ChatTemplateException("Unsupported subscript access")
        }
    }

    private fun sliceValue(base: Any?, start: Any?): Any? {
        if (base === Undefined || base == null) {
            return emptyList<Any?>()
        }

        val list = base as? List<*> ?: throw ChatTemplateException("Slicing is only supported for lists")
        val rawIndex = start?.let { intValue(it) } ?: 0
        val normalized = if (rawIndex >= 0) rawIndex else (list.size + rawIndex).coerceAtLeast(0)
        val clamped = normalized.coerceIn(0, list.size)
        return list.subList(clamped, list.size)
    }

    private fun call(callee: Expression, arguments: List<CallArgument>): Any? {
        return when (callee) {
            is Expression.Variable -> callBuiltin(callee.name, arguments)
            is Expression.Member -> callMethod(callee.name, evaluate(callee.base), arguments)
            else -> throw ChatTemplateException("Unsupported call expression")
        }
    }

    private fun callBuiltin(name: String, arguments: List<CallArgument>): Any? {
        return when (name) {
            "raise_exception" -> {
                val message = evaluate(requirePositionalArguments(arguments, 1).single())
                throw ChatTemplateException(stringValue(message))
            }
            "namespace" -> {
                val values = mutableMapOf<String, Any?>()
                for (argument in arguments) {
                    when (argument) {
                        is CallArgument.Named -> values[argument.name] = evaluate(argument.expression)
                        is CallArgument.Positional -> throw ChatTemplateException("namespace() only accepts named arguments")
                    }
                }
                Namespace(values)
            }
            "range" -> {
                val values = requirePositionalArguments(arguments, null).map { evaluate(it) }
                val bounds = when (values.size) {
                    1 -> 0 to (intValue(values[0]) ?: throw ChatTemplateException("range() expects integer arguments"))
                    2 -> {
                        val start = intValue(values[0]) ?: throw ChatTemplateException("range() expects integer arguments")
                        val end = intValue(values[1]) ?: throw ChatTemplateException("range() expects integer arguments")
                        start to end
                    }
                    else -> throw ChatTemplateException("range() expects one or two arguments")
                }
                (bounds.first until bounds.second).toList()
            }
            else -> throw ChatTemplateException("Unknown function: $name")
        }
    }

    private fun callMethod(name: String, base: Any?, arguments: List<CallArgument>): Any? {
        requirePositionalArguments(arguments, 0)
        val value = stringValue(base)

        return when (name) {
            "strip" -> value.trim()
            "title" -> value
                .lowercase()
                .split(Regex("\\s+"))
                .filter { it.isNotEmpty() }
                .joinToString(" ") {
                    it.replaceFirstChar { character -> character.titlecase(Locale.getDefault()) }
                }
            else -> throw ChatTemplateException("Unknown method: $name")
        }
    }

    private fun applyFilter(name: String, value: Any?): Any? {
        return when (name) {
            "trim" -> stringValue(value).trim()
            "length" -> when (value) {
                is String -> value.length
                is List<*> -> value.size
                is Map<*, *> -> value.size
                is Namespace -> value.values.size
                else -> 0
            }
            else -> throw ChatTemplateException("Unknown filter: $name")
        }
    }

    private fun requirePositionalArguments(arguments: List<CallArgument>, expected: Int?): List<Expression> {
        val positional = arguments.map {
            when (it) {
                is CallArgument.Positional -> it.expression
                is CallArgument.Named -> throw ChatTemplateException("Unexpected named argument: ${it.name}")
            }
        }

        if (expected != null && positional.size != expected) {
            throw ChatTemplateException("Expected $expected arguments")
        }

        return positional
    }

    private fun sequenceValues(value: Any?): List<Any?> {
        return if (value is List<*>) {
            value.toList()
        } else {
            emptyList()
        }
    }
}
