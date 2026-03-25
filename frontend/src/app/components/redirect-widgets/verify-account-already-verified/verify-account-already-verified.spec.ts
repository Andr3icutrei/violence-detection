import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerifyAccountAlreadyVerified } from './verify-account-already-verified';

describe('VerifyAccountAlreadyVerified', () => {
  let component: VerifyAccountAlreadyVerified;
  let fixture: ComponentFixture<VerifyAccountAlreadyVerified>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerifyAccountAlreadyVerified],
    }).compileComponents();

    fixture = TestBed.createComponent(VerifyAccountAlreadyVerified);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
