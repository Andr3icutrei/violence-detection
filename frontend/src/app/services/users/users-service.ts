import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment.development';
import { Observable } from 'rxjs';
import { UserResponseDto } from '../../core/api/models/user-response-dto';

@Injectable({
  providedIn: 'root',
})
export class UsersService {

  constructor(private httpClient: HttpClient) {

  }

  public register(email: string, password: string): Observable<UserResponseDto> {
    if(!email || !password){
      throw new Error('Email and password are required for registration.');
    }

    const body = {
      "email": email,
      "password": password
    };

    return this.httpClient.post<UserResponseDto>(environment.apiUrl + 'create', body);
  }
}
