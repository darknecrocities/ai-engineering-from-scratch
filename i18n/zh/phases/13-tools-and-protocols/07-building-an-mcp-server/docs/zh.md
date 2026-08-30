# 构建MCP服务器:无状态的Python和TypeScript

> 现代MCP服务器不记得握手. 它验证了每一个请求的元数据,运行一个处理器,并返回一个输入结果.

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 minutes

## 学习目标

- 执行强制性`server/discover`对于MCP`2026-07-28`现在,我们要去.
- 在每一个请求中验证协议版本和客户端功能.
- 通过确定性列表排序,将工具,资源和提示展示.
- 返回`resultType`服务器身份,以及对正确结果的缓存提示.
- 在Python和TypeScript中使用新线程有限的工作室使用相同的无国籍合同.

## 问题

服务器在第一次消息后存储客户端功能是容易构建的和难以操作的.同样的过程可能为序列客户端服务.远程请求可能会降落于不同的工作者.一个陈旧的功能声明可以泄露行为跨权限界限.

股`2026-07-28`您的应用程序仍然可以保存持久的笔记,工作或明确状态处理.它无法保留的是隐藏的协议状态,改变了后来的请求如何解码.

通过此课程,我们将两次构建一个笔记服务器.Python 和TypeScript版本仅使用其标准库用于协议核心.

## 概念

### 现代发送循环

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

工作室的三个规则仍然重要:

- 写JSON-RPC短信给STOUT,发送诊断给STODR.
- 通过新线来界定消息,然后将每个回复都填写在线.
- 快速出发,当STDIN到达EOF.

过程寿命是运输寿命,而不是现代的MCP会议.

### 申请验证

每个请求都必须包含:

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

需要使用前两种字段.`clientInfo`确认现有身份形状,但不要把它视为身份验证.

如果版本不支持,返回代码`-32022`随着`requested`其他`supported`错失的请求元数据是无效的参数,代码`-32602`永远不要填写之前的电话中缺失的字段.

### 必须发现

现代服务器必须实现`server/discover`完整的发现结果包括支持的现代版本,功能,可选指令,缓存提示和结果中的服务器身份`_meta`其他:

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

发现无法解锁服务器.`tools/list`没有说发现,因为`tools/list`已包含相同的请求元数据.

### 工具

`tools/list`稳定排序改善响应缓存并保持模型文本稳定.结果还需要`ttlMs`其他`cacheScope`现在,我们要去.

`tools/call`返回内容块和`isError`使用JSON-RPC错误,当协议包裹或方法参数不有效时. 使用 `isError: true`当有效的工具调用运行,但工具本身失败时.

工具注释仍然是提示,而不是执行:

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

服务器必须执行真正的授权.

### 资源

`resources/list`返回稳定URI描述符. `resources/read`输入内容. 两个都可在 `2026-07-28`两个都包括`ttlMs`其他`cacheScope`现在,我们要去.

使用`cacheScope: "private"`对于用户特定的注释数据. 分享缓存不能在授权环境中重复使用私人响应.

现代变更交付不使用`resources/subscribe`一个客户打开了`subscriptions/listen`要求`resourceSubscriptions`课程10建立了流量.

### 提示

`prompts/list`它们是可隐藏的,也是确定性的.`prompts/get`转换提示结果是完整的,但它不是可缓存列表或读取结果之一,需要缓存提示.

### 每个成功的结果都会被打字

例子中每次成功都用一个包装:

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

列表,阅读和发现处理人员添加`ttlMs`另外`cacheScope`集中包装可以防止一个处理器默默忽略现代结果场地.

### 没有服务器启动的请求

现代服务器可以发送与客户端请求相关的通知,或者在客户端开放的通知.`subscriptions/listen`它们不能发送自己的JSON-RPC请求.

当处理器需要采样,引发或根输入时,它返回一个`input_required`结果. 客户端完成嵌入式输入请求,并用新的请求ID重新尝试原始方法. 第11课涵盖了多轮访问请求模式.

### 显而易见的遗产兼容性

双代服务器也可以实现`2025-11-25`它们在一个明显分离的遗产分支上握手.`_meta`收到时的现象和遗产行为`initialize`现在,我们要去.

不要放一个`2026-07-28`通过传统的握手路径来请求.`resultType`在本课程中,代码是故意现代化的,所以其变量保持可见.

```figure
t3-dispatch-loop
```

## 用它

运行Python服务器的有限演示和测试:

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

使用TypeScript运行器运行TypeScript端口:

```bash
npx tsx main.ts --demo
```

演示器发送了`server/discover`现在,我们可以看到一个版本错误,然后我们可以看到一个版本错误,然后我们可以看到一个版本错误.

## 运送它

这一课是很好的.`outputs/skill-mcp-server-scaffolder.md`它提供了一个现代化的服务器计划,包括一个发现合同,每次请求验证,确定性可缓存列表和可选的孤立遗产适配器.

## 运动

1. 删除一个请求的功能,证明服务器不重复使用前一个请求的声明.
2. 扭转`TOOLS`现在`PROMPTS`确认所有列表结果保持稳定.
3. 添加一个破坏性的`notes_delete`执行器内进行授权检查.`destructiveHint`只是一个 UX 暗示.
4. 加入`resources/templates/list`随着`ttlMs`现在`cacheScope`它们是指"定性定制"的.
5. 建立一个独立的传统适配器`2025-11-25`添加测试证明现代请求从来没有进入.

## 关键词

| Term | Meaning |
|------|---------|
| Stateless server | Handles each request from its own metadata without protocol-session memory |
| `server/discover` | Mandatory modern method that advertises versions and capabilities |
| Complete result | Successful modern result with `resultType: "complete"` |
| Cacheable result | Discovery, list, or resource-read result with `ttlMs` and `cacheScope` |
| Deterministic list | Same logical registry produces the same item order |
| Server identity | Recommended `io.modelcontextprotocol/serverInfo` in result `_meta` |
| Tool error | Valid tool call that returns content with `isError: true` |
| Protocol error | Invalid JSON-RPC or MCP request returned through `error` |

## 进一步阅读

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
