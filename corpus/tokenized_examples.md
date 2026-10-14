# Three tokenized examples

## methods (joml-ci/joml, 70 tokens)

Text:

```
/** Set each component of this vector to the largest (closest to positive infinity) {@code double} value that is less than or equal to that component and is equal to a mathematical integer. @return this */ public Vector3d floor() { this.x = Math.floor(x); this.y = Math.floor(y); this.z = Math.floor(z); return this; }
```

Pieces (`▁` marks a preceding space):

```
▁/** | ▁Set | ▁each | ▁component | ▁of | ▁this | ▁vector | ▁to | ▁the | ▁largest | ▁(c | lose | st | ▁to | ▁positive | ▁in | finity | ) | ▁{@code | ▁double | } | ▁value | ▁that | ▁is | ▁less | ▁than | ▁or | ▁equal | ▁to | ▁that | ▁component | ▁and | ▁is | ▁equal | ▁to | ▁a | ▁m | ath | em | at | ical | ▁inte | ger. | ▁@return | ▁this | ▁*/ | ▁public | ▁Vector3d | ▁floor | () | ▁{ | ▁this.x | ▁= | ▁Math.floor | (x | ); | ▁this.y | ▁= | ▁Math.floor | (y | ); | ▁this.z | ▁= | ▁Math.floor | (z | ); | ▁return | ▁this; | ▁} | </s>
```

## commits (joml-ci/joml, 22 tokens)

Text:

```
Rename Quaternion.lookRotate() to .lookAlong() This is now equivalent to the Matrix3/4 methods.
```

Pieces (`▁` marks a preceding space):

```
▁Rename | ▁Quaternion | . | look | Rotate | () | ▁to | ▁. | look | Along | () | ▁This | ▁is | ▁now | ▁equivalent | ▁to | ▁the | ▁Matrix | 3 | /4 | ▁methods. | </s>
```

## reviews (hernanponcedeleon/dat3m, 99 tokens)

Text:

```
<code>@@ -109,8 +109,10 @@ public Type visitOpTypeStruct(SpirvParser.OpTypeStructContext ctx) { return builder.addType(id, type); } throw new ParsingException("Illegal member offset decorations for struct '%s'", id); + } else { + Type type = types.getAggregateType(memberTypes); + return builder.addType(id, type); }</code><technical_language>The "else" wrapper can be removed, i.e. throw new ParsingException("Illegal member offset decorations for struct '%s'", id); } Type type = types.getAggregateType(memberTypes); return builder.addType(id, type); Removed</technical_language>
```

Pieces (`▁` marks a preceding space):

```
<code> | @@ | ▁-10 | 9,8 | ▁+10 | 9,10 | ▁@@ | ▁public | ▁Type | ▁visit | Op | Type | Struct | ( | SpirvParser. | Op | Type | Struct | Context | ▁ctx) | ▁{ | ▁return | ▁builder.add | Type( | id, | ▁type); | ▁} | ▁throw | ▁new | ▁ParsingException(" | Illegal | ▁member | ▁offset | ▁decor | ations | ▁for | ▁struct | ▁'%s'", | ▁id); | ▁+ | ▁} | ▁else | ▁{ | ▁+ | ▁Type | ▁type | ▁= | ▁types.get | Aggregate | Type | (member | Types); | ▁+ | ▁return | ▁builder.add | Type( | id, | ▁type); | ▁} | </code> | <technical_language> | The | ▁" | else | " | ▁wrapper | ▁can | ▁be | ▁removed, | ▁i.e. | ▁throw | ▁new | ▁ParsingException(" | Illegal | ▁member | ▁offset | ▁decor | ations | ▁for | ▁struct | ▁'%s'", | ▁id); | ▁} | ▁Type | ▁type | ▁= | ▁types.get | Aggregate | Type | (member | Types); | ▁return | ▁builder.add | Type( | id, | ▁type); | ▁Removed | </technical_language> | </s>
```
