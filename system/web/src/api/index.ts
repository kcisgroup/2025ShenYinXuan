// import { mockEventStreamText } from '@/data'
// import { currentHost } from '@/utils/location'
// import request from '@/utils/request'

/**
 * Event Stream 调用大模型接口 Ollama3 (Fetch 调用)
 */
export async function createOllama3Stylized(text, qa_type, uuid) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const businessStore = useBusinessStore()
    const url = new URL(`${location.origin}/sanic/dify/get_answer`)
    const params = {}
    Object.keys(params).forEach((key) => {
        url.searchParams.append(key, params[key])
    })

    //文件问答传文件url
    if (text.includes('总结归纳文档的关键信息')) {
        text = `${businessStore.$state.file_url}|总结归纳文档的关键信息`
    } else if (qa_type === 'FILEDATA_QA') {
        //表格问答默认带上文件url/key
        text = `${businessStore.$state.file_url}|${text}`
    }

    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({
            query: text,
            qa_type: qa_type,
            chat_id: uuid
        })
    })
    return fetch(req)
}

/**
 * 用户登录
 * @param username
 * @param password
 * @returns
 */
export async function login(username, password) {
    const url = new URL(`${location.origin}/sanic/user/login`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username,
            password
        })
    })
    return fetch(req)
}

/**
 * 查询用户对话历史
 * @param page
 * @param limit
 * @returns
 */
export async function query_user_qa_record(page, limit) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/user/query_user_record`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            page,
            limit
        })
    })
    return fetch(req)
}

/**
 * 删除对话历史记录
 * @param page
 * @param limit
 * @returns
 */
export async function delete_user_record(ids) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/user/delete_user_record`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            record_ids: ids
        })
    })
    return fetch(req)
}

/**
 * 用户反馈
 * @param chat_id
 * @param rating
 * @returns
 */
export async function fead_back(chat_id, rating) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/user/dify_fead_back`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            chat_id,
            rating
        })
    })
    return fetch(req)
}

/**
 * 问题建议
 * @param chat_id
 * @param rating
 * @returns
 */
export async function dify_suggested(chat_id) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/dify/get_dify_suggested`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            chat_id
        })
    })
    return fetch(req)
}

/**
 * word 转 md
 * @param file_key
 * @returns
 */
export async function word_to_md(file_key) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/ta/word_to_md`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            file_key
        })
    })
    return fetch(req)
}

/**
 * 查询项目列表
 * @param page
 * @param limit
 * @returns
 */
export async function query_demand_records(page, limit) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/ta/query_demand_records`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            page,
            limit
        })
    })
    return fetch(req)
}

/**
 * 保存项目信息
 * @param project_data
 * @returns
 */
export async function insert_demand_manager(project_data) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/ta/insert_demand_manager`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            project_data
        })
    })
    return fetch(req)
}

/**
 * 删除项目信息
 * @param id
 * @returns
 */
export async function delete_demand_records(id) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/ta/delete_demand_records`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            id
        })
    })
    return fetch(req)
}

/**
 * 抽取功能点
 * @param doc_id
 * @returns
 */
export async function abstract_doc_func(doc_id) {
    const userStore = useUserStore()
    const token = userStore.getUserToken()
    const url = new URL(`${location.origin}/sanic/ta/abstract_doc_func`)
    const req = new Request(url, {
        mode: 'cors',
        method: 'post',
        headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}` // 添加 token 到头部
        },
        body: JSON.stringify({
            doc_id
        })
    })
    return fetch(req)
}
