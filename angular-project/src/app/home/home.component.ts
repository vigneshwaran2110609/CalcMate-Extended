import { Component, ElementRef, ViewChild } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { ChangeDetectorRef } from '@angular/core';
import { HttpParams } from '@angular/common/http';

interface Chat {
  id: number;
  name: string;
  messages: { sender: 'user' | 'ai'; message: string }[];
  timestamp?: Date;
  isHeader?: boolean;
  selectedModel?: 'GPT' | 'Llama'
}

interface ChatResponse {
  chats: number;
  timestamps: {[id: string]: string};
}

@Component({
  selector: 'app-home',
  standalone: false,
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css'],
})
export class HomeComponent {
  problem: string = ''; // Input field for the problem
  loading: boolean = false; // Loading indicator
  taglineVisible: boolean = true; // Controls tagline visibility
  logoVisible: boolean = true; // Controls logo visibility
  responseFetched: boolean = false; // Tracks if a response was fetched
  inputVisible: boolean = true;
  loadingChat:boolean=false; // Controls visibility of input field
  selectedModel: 'GPT' | 'Llama' = 'GPT'; // Selected model
  isFirstQuestion: boolean = true;
  chatId: number | undefined;
  chi: number | undefined;
  fullText: string = 'Get your doubts cleared in no time!';
  displayedText: string = ''; // Text being typed
  currentCharIndex: number = 0;
  typingSpeed: number = 100; // Typing speed in ms

  chats: Chat[] = [];
  
  activeChatId: number | null = null;

  @ViewChild('chatContainer', { static: false }) chatContainer!: ElementRef;

  constructor(private http: HttpClient,private cdRef: ChangeDetectorRef) {}

  ngOnInit(): void {
    this.typeText();
    this.displayInitialChats();
    this.addChat();
  }

  typeText() {
    if (this.currentCharIndex < this.fullText.length) {
      this.displayedText += this.fullText.charAt(this.currentCharIndex);
      this.currentCharIndex++;
      setTimeout(() => this.typeText(), this.typingSpeed);
    }
  }

  addChat() {
    this.http.get<{chats: number}>(
      'http://localhost:5000/get-all-chats', 
      { withCredentials: true }
    ).subscribe(
      (response) => {
        this.chi = response.chats;
        
        // Create new chat with current timestamp
        const newChat: Chat = {
          id: this.chi + 1,
          name: `Word Problem ${this.chi + 1}`,
          messages: [],
          timestamp: new Date() // Add timestamp for the new chat
        };
        
        // Find the index of the "Today" header if it exists
        const todayHeaderIndex = this.chats.findIndex(chat => 
          chat.isHeader && chat.name === 'Today'
        );
        
        if (todayHeaderIndex !== -1) {
          // Insert the new chat right after the "Today" header
          this.chats.splice(todayHeaderIndex + 1, 0, newChat);
        } else {
          // If "Today" header doesn't exist, add it and then the new chat
          this.chats.unshift({ id: -1, name: 'Today', isHeader: true, messages: [] });
          this.chats.splice(1, 0, newChat);
        }
        
        // Set the new chat as active
        this.activeChatId = newChat.id;
        console.log(this.activeChatId);
        console.log(newChat.id);
        
        // Notify the server about the new chat
        this.http.post(
          'http://localhost:5000/chat-started', 
          { chat_id: newChat.id },
          { withCredentials: true }
        ).subscribe(
          () => console.log(`Word Problem ${newChat.id} started successfully`),
          (error) => console.error('Error sending chat ID:', error)
        );
      }
    );
  }
  
  displayInitialChats() {
    this.http.get<ChatResponse>(
      'http://localhost:5000/get-all-chats',
      { withCredentials: true }
    ).subscribe(
      (response) => {
        this.chatId = response.chats;
        console.log("Here 3");
        console.log(this.chatId);
        
        // Create categorized chat arrays
        const todayChats: Chat[] = [];
        const yesterdayChats: Chat[] = [];
        const olderChats: Chat[] = [];
        let mostRecentChatId: number | null = null;
        let mostRecentTimestamp: Date = new Date(0); // Initialize with oldest possible date
        
        if (this.chatId !== null) {
          // Get current date (at the start of the day)
          const today = new Date();
          today.setHours(0, 0, 0, 0);
          
          // Get yesterday date
          const yesterday = new Date(today);
          yesterday.setDate(yesterday.getDate() - 1);
          
          // Get date 7 days ago
          const sevenDaysAgo = new Date(today);
          sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
          
          for (let i = 1; i <= this.chatId; i++) {
            // Create chat object with properly typed messages array
            const chatTimestamp = response.timestamps[i.toString()] 
              ? new Date(response.timestamps[i.toString()]) 
              : new Date();
              
            const chat: Chat = {
              id: i,
              name: `Word Problem ${i}`,
              messages: [],
              timestamp: chatTimestamp
            };
            
            // Track the most recent chat
            if (chatTimestamp > mostRecentTimestamp) {
              mostRecentTimestamp = chatTimestamp;
              mostRecentChatId = i;
            }
            
            // Categorize the chat based on its timestamp
            const chatDate = new Date(chatTimestamp);
            chatDate.setHours(0, 0, 0, 0);
            
            if (chatDate.getTime() === today.getTime()) {
              todayChats.push(chat);
            } else if (chatDate.getTime() === yesterday.getTime()) {
              yesterdayChats.push(chat);
            } else if (chatDate.getTime() >= sevenDaysAgo.getTime()) {
              olderChats.push(chat);
            } else {
              olderChats.push(chat);
            }
          }
          
          // Sort each category by timestamp in descending order (newest first)
          todayChats.sort((a, b) => {
            return new Date(b.timestamp as Date).getTime() - new Date(a.timestamp as Date).getTime();
          });
          
          yesterdayChats.sort((a, b) => {
            return new Date(b.timestamp as Date).getTime() - new Date(a.timestamp as Date).getTime();
          });
          
          olderChats.sort((a, b) => {
            return new Date(b.timestamp as Date).getTime() - new Date(a.timestamp as Date).getTime();
          });
          
          // Clear existing chats array
          this.chats = [];
          
          // Add section headers and combine all chats
          if (todayChats.length > 0) {
            this.chats.push({ id: -1, name: 'Today', isHeader: true, messages: [] });
            this.chats = this.chats.concat(todayChats);
          }
          
          if (yesterdayChats.length > 0) {
            this.chats.push({ id: -2, name: 'Yesterday', isHeader: true, messages: [] });
            this.chats = this.chats.concat(yesterdayChats);
          }
          
          if (olderChats.length > 0) {
            this.chats.push({ id: -3, name: 'Older', isHeader: true, messages: [] });
            this.chats = this.chats.concat(olderChats);
          }
          
          // Set the most recent chat as active if available
          if (mostRecentChatId !== null) {
            this.selectChat(mostRecentChatId);
          }
        }
        
        console.log(this.chats);
      },
      (error) => {
        console.error('Error fetching initial chats:', error);
      }
    );
  }

  selectChat(chatId: number) {
    this.activeChatId = chatId;
    this.loadingChat = true;
    this.taglineVisible = false;
    this.logoVisible = false;
    this.inputVisible = false;
    this.isFirstQuestion = true;
    this.cdRef.detectChanges(); 

    this.http.post<{ problem: string[]; solution: string[] }>(
      'http://localhost:5000/get-chat-history',
      { chat_id: chatId } , { withCredentials: true }
    ).subscribe(
      (response) => {
        this.loadingChat = false;

        if (response.problem?.length && response.solution?.length) {
         
          this.taglineVisible = false;
          this.logoVisible = false;
          this.inputVisible = true;

          const chat = this.chats.find(c => c.id === chatId);
          if (chat) {
         
            chat.messages = [];
            const maxLength = Math.max(response.problem.length, response.solution.length);

            for (let i = 0; i < maxLength; i++) {
              if (i < response.problem.length) {
                chat.messages.push({ sender: 'user', message: response.problem[i] });
              }
              if (i < response.solution.length) {
                chat.messages.push({ sender: 'ai', message: response.solution[i] });
              }
            }
          }
        } else {
        
          this.taglineVisible = true;
          this.logoVisible = true;
          this.inputVisible = true;
        }

        this.cdRef.detectChanges(); 
      },
      (error) => {
        this.loadingChat = false; 
        console.error('Error fetching chat history:', error);
      }
    );
  }

  getMessagesForChat() {
    const chat = this.chats.find(c => c.id === this.activeChatId);
    return chat ? chat.messages : [];
  }

  solve(problemType: 'GPT' | 'Llama') {
    if (this.problem.trim() && this.activeChatId !== null) {  // Ensure a chat is selected
      this.taglineVisible = false;
      this.logoVisible = false;
      this.isFirstQuestion = false;
      this.selectedModel = problemType;
  
      const activeChat = this.chats.find(chat => chat.id === this.activeChatId);
      if (!activeChat) return;
  
      activeChat.messages.push({ sender: 'user', message: this.problem });
  
      this.loading = true;
      const endpoint =
        problemType === 'GPT'
          ? 'http://localhost:5000/solve-gpt'
          : 'http://localhost:5000/solve-llama';
  
      this.http.post<{ solution: string }>(
        endpoint,
        { 
          problem: this.problem, 
          chat_id: this.activeChatId,
          responseFetched: this.responseFetched 
        },
        { withCredentials: true }
      ).subscribe(
        (response) => {
          console.log("Hello from backend");
          activeChat.messages.push({ sender: 'ai', message: response.solution });
          this.loading = false;
          this.responseFetched = true;
          this.problem = '';
          this.inputVisible = true;
          setTimeout(() => this.scrollToBottom(), 100);
        },
        (error) => {
          activeChat.messages.push({
            sender: 'ai',
            message: 'Error while fetching solution.',
          });
          this.loading = false;
          this.inputVisible = true;
          setTimeout(() => this.scrollToBottom(), 100);
        }
      );
    }
  }

  download() {
    if (this.activeChatId !== null){
      const params = new HttpParams().set('chat_id', this.activeChatId.toString());
      this.http.get('http://localhost:5000/download', { responseType: 'blob' , withCredentials: true, params: params }).subscribe(
        (response) => {
          const blob = new Blob([response], { type: 'application/pdf' });
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'calcmate_solutions.pdf';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          alert("Downloaded File!!!");
        },
        (error) => {
          alert("Cannot download empty file");
          
        }
      );
    }
  }

  scrollToBottom() {
    requestAnimationFrame(() => {
      if (this.chatContainer && this.chatContainer.nativeElement) {
        this.chatContainer.nativeElement.scrollTop =
          this.chatContainer.nativeElement.scrollHeight;
      }
    });
  }
}